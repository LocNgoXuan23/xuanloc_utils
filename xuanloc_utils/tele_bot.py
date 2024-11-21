import asyncio
import telegram

class TeleBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.tele_bot = telegram.Bot(token=token)
        
    def send_message(self, message):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tele_bot.send_message(chat_id=self.chat_id, text=message))
        print(f"Message sent: {message}")
        