import logging
import re
import requests
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

logger = logging.getLogger(__name__)

# URL for telebirr receipt verification
TELEBIRR_VERIFICATION_URL = "https://transactioninfo.ethiotelecom.et/receipt/{}"
# The string to check for on a successful verification page
SUCCESS_STRING = "የቴሌብር ክፍያ መረጃ/telebirr Transaction information"

# Initialize the docTR OCR model once when the module is imported
ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def extract_transaction_info(text_data):
    """
    Parses the text data to find "Transaction Number" and extracts the value that follows.
    """
    label = "Transaction Number"
    
    text_data_lower = text_data.lower()
    
    if label.lower() in text_data_lower:
        try:
            start_index = text_data_lower.find(label.lower())
            end_of_label = start_index + len(label)
            
            # Slice the string from the end of the label to the end of the document
            remainder = text_data[end_of_label:].strip().replace(':', '')
            
            if remainder:
                tx_id = remainder.split()[0].upper()
                return tx_id
        except (IndexError, ValueError) as e:
            logger.warning(f"Error extracting transaction ID: {e}")
            
    return None

async def process_image_for_txid(image_path: str):
    """
    Processes a single image for a transaction ID using OCR.
    """
    try:
        logger.info(f"Starting docTR OCR on image: {image_path}")
        doc = DocumentFile.from_images([image_path])
        result = ocr_model(doc)
        
        if not result or not result.pages:
            logger.warning("docTR did not return any pages. OCR likely failed.")
            return None, None
            
        full_text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    full_text += " ".join([word.value for word in line.words]) + "\n"
        
        if not full_text.strip():
            logger.warning("docTR extracted empty text. OCR likely failed to detect anything.")
            return None, None

        logger.info(f"Extracted text using docTR: \n{full_text}")

        tx_id = extract_transaction_info(full_text)
        
        return tx_id, TELEBIRR_VERIFICATION_URL
        
    except Exception as e:
        logger.error(f"Error processing image {image_path} with docTR: {e}", exc_info=True)
        return None, None

def verify_telebirr_receipt(tx_id: str, verify_url: str) -> bool:
    """
    Verifies a Telebirr receipt by making an HTTP request.
    This function includes OCR correction logic.
    """
    full_url = verify_url.format(tx_id)
    try:
        response = requests.get(full_url, timeout=10)
        
        if response.status_code == 200 and SUCCESS_STRING in response.text:
            return True
        
        # OCR correction logic
        if 'O' in tx_id:
            normalized_tx_id = tx_id.replace('O', '0')
            normalized_url = verify_url.format(normalized_tx_id)
            normalized_response = requests.get(normalized_url, timeout=10)
            if normalized_response.status_code == 200 and SUCCESS_STRING in normalized_response.text:
                return True
        
        if '0' in tx_id:
            normalized_tx_id = tx_id.replace('0', 'O')
            normalized_url = verify_url.format(normalized_tx_id)
            normalized_response = requests.get(normalized_url, timeout=10)
            if normalized_response.status_code == 200 and SUCCESS_STRING in normalized_response.text:
                return True
        
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during HTTP request: {e}")
        return False