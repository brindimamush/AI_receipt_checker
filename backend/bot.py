import logging
import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from dotenv import load_dotenv
# Set up logging for better error handling and debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
load_dotenv()
# Replace with your actual bot token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize the docTR OCR model.
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

async def start(update: Update, context) -> None:
    await update.message.reply_text('Hello! Send me a picture of a Telebirr receipt and I will check it for you.')

def extract_transaction_info(text_data):
    """
    Parses the text data to find "Transaction Number" and extracts the value that follows.
    """
    label = "Transaction Number"
    verify_url = "https://transactioninfo.ethiotelecom.et/receipt/{}"
    bank_name = "Telebirr"
    
    text_data_lower = text_data.lower()
    
    if label.lower() in text_data_lower:
        try:
            start_index = text_data_lower.find(label.lower())
            end_of_label = start_index + len(label)
            
            # Slice the string from the end of the label to the end of the document
            remainder = text_data[end_of_label:].strip().replace(':', '')
            
            if remainder:
                tx_id = remainder.split()[0].upper()
                return bank_name, tx_id, verify_url
        except (IndexError, ValueError) as e:
            logger.warning(f"Error extracting transaction ID: {e}")
            
    return None, None, None

async def process_image_for_txid(image_path: str):
    """
    Helper function to process a single image for transaction ID.
    """
    try:
        logger.info(f"Starting docTR OCR on image: {image_path}")
        doc = DocumentFile.from_images([image_path])
        result = model(doc)
        
        if not result or not result.pages:
            logger.warning("docTR did not return any pages. OCR likely failed.")
            return None, None, None
            
        full_text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    full_text += " ".join([word.value for word in line.words]) + "\n"
        
        if not full_text.strip():
            logger.warning("docTR extracted empty text. OCR likely failed to detect anything.")
            return None, None, None

        logger.info(f"Extracted text using docTR: \n{full_text}")

        bank_name, tx_id, verify_url = extract_transaction_info(full_text)
        
        return bank_name, tx_id, verify_url
        
    except Exception as e:
        logger.error(f"Error processing image {image_path} with docTR: {e}", exc_info=True)
        return None, None, None

async def handle_photo(update: Update, context) -> None:
    photo_file = await update.message.photo[-1].get_file()
    file_path = f'downloads/{photo_file.file_id}.jpg'
    os.makedirs('downloads', exist_ok=True)
    await photo_file.download_to_drive(file_path)

    await update.message.reply_text("Processing your receipt...")
    
    bank_name, tx_id, verify_url = await process_image_for_txid(file_path)
    
    if tx_id and verify_url:
        full_url = verify_url.format(tx_id)
        
        try:
            # First attempt with the raw OCR result
            response = requests.get(full_url, timeout=10)
            
            # The HTML snippet you provided is the key to our success check
            success_string = "የቴሌብር ክፍያ መረጃ/telebirr Transaction information"
            
            if response.status_code == 200 and success_string in response.text:
                await update.message.reply_text("✅ Congratulations! The receipt is valid. 🎉")
                await update.message.reply_text(f"You can view the full receipt here: {full_url}")
            else:
                # If the first attempt fails, try specific OCR corrections
                
                # Try replacing 'O' with '0'
                if 'O' in tx_id:
                    normalized_tx_id = tx_id.replace('O', '0')
                    normalized_url = verify_url.format(normalized_tx_id)
                    normalized_response = requests.get(normalized_url, timeout=10)
                    if normalized_response.status_code == 200 and success_string in normalized_response.text:
                        await update.message.reply_text("✅ Congratulations! The receipt is valid after correcting an OCR error (O to 0). 🎉")
                        await update.message.reply_text(f"You can view the full receipt here: {normalized_url}")
                        # Return to avoid further checks
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        return
                        
                # Try replacing '0' with 'O' if the previous attempt failed
                if '0' in tx_id:
                    normalized_tx_id = tx_id.replace('0', 'O')
                    normalized_url = verify_url.format(normalized_tx_id)
                    normalized_response = requests.get(normalized_url, timeout=10)
                    if normalized_response.status_code == 200 and success_string in normalized_response.text:
                        await update.message.reply_text("✅ Congratulations! The receipt is valid after correcting an OCR error (0 to O). 🎉")
                        await update.message.reply_text(f"You can view the full receipt here: {normalized_url}")
                        # Return to avoid further checks
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        return
                        
                # If all attempts fail
                await update.message.reply_text("❌ The receipt could not be verified. It appears to be invalid or there was an OCR error.")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during HTTP request: {e}")
            await update.message.reply_text("An error occurred while trying to verify the receipt. Please try again later.")
    else:
        await update.message.reply_text("Could not find a valid transaction ID in the receipt. Please try again with a clearer image.")
        
    if os.path.exists(file_path):
        os.remove(file_path)

async def handle_document(update: Update, context) -> None:
    await update.message.reply_text("This bot only processes images for Telebirr receipts.")

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))
    application.add_handler(MessageHandler(filters.Document.PDF & ~filters.COMMAND, handle_document))
    application.run_polling()

if __name__ == "__main__":
    main()