import speech_recognition as sr

class SpeechDetector():
    def __init__(self):
        self.listener = sr.Recognizer()
    
    def getAudio(self):
        with sr.Microphone() as source:
            try:
                print("listening...")
                voice = self.listener.listen(source)
                data = self.listener.recognize_google(voice)
                print(data)
                return data
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")