import re

GENERIC_RESPONSES = {
    # greetings
    "hi": "Hello.",
    "hello": "Hello.",
    "hey": "Hello.",
    "good morning": "Good morning. How can I assist you?",
    "good afternoon": "Good afternoon. What can I help you with?",
    "good evening": "Good evening. How may I assist you?",
    "hi there": "hi",
    "hello there": "hello",
    "helo": "hello",
    "hey there": "hey",

    # farewells
    "bye": "Goodbye. Have a great day.",
    "goodbye": "Goodbye. Take care.",
    "see you": "Goodbye.",
    "see ya": "Goodbye.",
    "take care": "Goodbye.",
    "cya": "bye",

    # gratitude
    "thank you": "You're welcome.",
    "thanks": "You're welcome.",
    "thank u": "You're welcome.",
    "ty": "You're welcome.",
    "thx": "You're welcome.",
    "appreciate it": "You're welcome.",

    # small talk / identity
    "how are you": "I'm functioning well, thank you for asking.",
    "who are you": "I'm the UNH Catalog Assistant, here to help you find academic and program information.",
    "what is your purpose": "I'm designed to answer questions about UNH programs, courses, and admissions.",
    "what can you do": "I can provide information about programs, courses, requirements, and policies at UNH.",
    "can you help me": "Certainly. What would you like help with?",
    "i need help": "Of course. What are you looking for?",
    "help": "Sure. Please tell me what you’d like to know about UNH.",
    "not sure": "That’s okay. Could you tell me a bit more about what you’re looking for?",
    "i don’t know": "No problem. I can help you find out.",
    "idk": "No problem. I can help you find out.",

    # confirmation / clarification
    "what do you mean": "Let me clarify that for you.",
    "wait": "No problem. Take your time.",
    "one sec": "Sure, I’ll be here.",
    "hold on": "Of course.",

    # neutral feedback / short replies
    "yes": "Understood.",
    "no": "Okay.",
    "maybe": "Understood.",
    "fine": "Alright.",
    "good": "Glad to hear it.",
    "great": "Good to know.",
    "perfect": "Excellent.",
    "understood": "Got it.",
    "that’s fine": "Okay.",
    "sounds good": "Alright.",

    # apologies / politeness
    "sorry": "That’s okay.",
    "sorry about that": "No problem.",
    "my bad": "That’s alright.",
    "no problem": "Glad to assist.",

    # courtesy / social
    "nice to meet you": "Nice to meet you as well.",
    "pleased to meet you": "Nice to meet you.",
    "thank you very much": "You're welcome.",
    "thank you so much": "You're welcome.",
}

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def get_generic_response(message: str) -> str | None:
    text = normalize_text(message)
    matches = []

    for key, reply in GENERIC_RESPONSES.items():
        if not reply:
            continue
        for m in re.finditer(rf"\b{re.escape(key)}\b", text):
            matches.append((m.start(), m.end(), len(key), reply))

    if not matches:
        return None

    # sort by start position, then by key length (longest first)
    matches.sort(key=lambda x: (x[0], -x[2]))

    responses = []
    used_spans = []

    for start, end, _, reply in matches:
        if any(s <= start <= e or s <= end <= e for s, e in used_spans):
            continue
        responses.append(reply)
        used_spans.append((start, end))

    if responses:
        return " ".join(responses)
    return None