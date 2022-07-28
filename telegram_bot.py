from telegram.ext import *
from io import BytesIO
import numpy as np
import PIL.Image as Image
from model import load_model
import os
from torchvision import transforms as T


# Bot token
TOKEN = '5515046597:AAELNS1jjaxnWHZMiIXJEz9VHCjY2snxJSY'

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = load_model(os.path.join(BASE_DIR, 'models', 'model_v1.pt'))

# Set classes
classes = ['Australian terrier',
            'Border terrier',
            'Samoyed',
            'Beagle',
            'Shih-Tzu',
            'English foxhound',
            'Rhodesian ridgeback',
            'Dingo',
            'Golden retriever',
            'Old English sheepdog']


# Start conversation
def start(update, context):
    update.message.reply_text('Welcome!')

# Help message
def help(update, context):
    update.message.reply_text("""
    Upload picture to get prediction
    
    /start - Starts conversation
    /help - Show help
    
    """)

# Instruction message
def handle_message(update, context):
    update.message.reply_text('Please send a picture')


# Handle photo
def handle_photo(update, context):

    # Get last sent photo as stream of bytes
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.array((bytearray(f.read())), dtype=np.uint8)

    # Get image from bytestream
    image = Image.open(BytesIO(file_bytes))

    # Image transforms
    transforms = T.Compose([T.Resize(size=(224, 224)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])
    # Make image RGB
    if len(np.array(image).shape):
        image = image.convert('RGB')

    # Transform image
    image = transforms(image)

    # get predictions
    pred, predict_proba = model.predict(image)

    # Get best prediction
    top_prob_class = sorted(zip(np.around(np.array(predict_proba[0]), 2) * 100, classes), reverse=True)[:1]

    # Write message with prediction
    update.message.reply_text('In this picture i see: {}\nCertainty: {:.0f} %'.format(top_prob_class[0][1], top_prob_class[0][0]))



updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

# Command function handler
dp.add_handler(CommandHandler('start', start))
dp.add_handler(CommandHandler('help', help))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()