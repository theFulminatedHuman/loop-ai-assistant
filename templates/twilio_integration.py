from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import requests
import os

app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/twilio/voice', methods=['POST'])
def handle_twilio_voice():
    """Handle incoming Twilio voice calls"""
    response = VoiceResponse()
    
    # Introduction
    response.say("Hello! Welcome to Loop AI Hospital Network Assistant. I can help you find hospitals in our network and verify coverage.", 
                 voice='alice')
    
    # Gather user input
    gather = Gather(
        input='speech',
        action='/twilio/process_speech',
        method='POST',
        speech_timeout=2
    )
    gather.say("How can I help you today? For example, you can say 'Find hospitals in Bangalore' or 'Is Manipal Hospital in my network?'")
    response.append(gather)
    
    # If no input, redirect to same endpoint
    response.redirect('/twilio/voice')
    
    return str(response)

@app.route('/twilio/process_speech', methods=['POST'])
def process_speech():
    """Process speech input from Twilio"""
    speech_result = request.form.get('SpeechResult', '')
    response = VoiceResponse()
    
    if speech_result:
        # Call our assistant API
        api_response = requests.post('http://localhost:5000/api/voice', json={
            'transcript': speech_result,
            'session_id': request.form.get('CallSid')
        })
        
        if api_response.status_code == 200:
            data = api_response.json()
            response.say(data['response'], voice='alice')
            
            if not data.get('end_conversation', False):
                # Continue conversation
                gather = Gather(
                    input='speech',
                    action='/twilio/process_speech',
                    method='POST',
                    speech_timeout=2
                )
                gather.say("Is there anything else I can help you with?")
                response.append(gather)
        else:
            response.say("I'm sorry, I'm having technical difficulties. Please try again later.", voice='alice')
    else:
        response.say("I didn't catch that. Please try again.", voice='alice')
        response.redirect('/twilio/voice')
    
    return str(response)

def purchase_phone_number():
    """Purchase a Twilio phone number (run once)"""
    try:
        numbers = client.available_phone_numbers('US').local.list(limit=1)
        if numbers:
            phone_number = client.incoming_phone_numbers.create(
                phone_number=numbers[0].phone_number,
                voice_url=f"https://your-domain.com/twilio/voice"
            )
            print(f"Purchased number: {phone_number.phone_number}")
    except Exception as e:
        print(f"Error purchasing number: {e}")

if __name__ == '__main__':
    app.run(port=5001)
