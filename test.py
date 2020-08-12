import pyttsx3

engine = pyttsx3.init()
rate = engine.getProperty('rate')
print(rate)
engine.setProperty('rate', rate + 200)
engine.say('你好')
engine.runAndWait()

# engine.setProperty('voice', 'zh')
# engine.say('Sally sells seashells by the seashore.')
