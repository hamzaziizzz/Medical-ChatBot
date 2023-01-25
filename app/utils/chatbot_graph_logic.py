import json

from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer, text_to_word_sequence

from app.utils.answer_search import *
from app.utils.question_classifier import *
from app.utils.question_parser import *


# ============================ GUI IMPORTS ============================
import time
from tkinter import *
import tkinter.messagebox
import pyttsx3

saved_username = ["You"]
window_size = "800x800"


# ============================ SIAMESE MODEL ============================
def load_siamese_model():
    json_file = open("..\\app\\models\\model.json")
    loaded_model_json = json_file.read()
    json_file.close()

    # load model
    model = model_from_json(loaded_model_json)

    # load weights
    model.load_weights("..\\app\\models\\question_pairs_weights.h5")

    return model

def convert_text_to_index_array(text, dictionary):
    words = text_to_word_sequence(text)
    word_indices = []
    for word in words:
        if word in dictionary:
            word_indices.append(dictionary[word])
        else:
            print(f"'{word}' not in training corpus; ignoring.")

    return word_indices

def find_if_duplicate_questions(question_1, question_2):
    tokenizer = Tokenizer(num_words=100000)

    with open("..\\app\\models\\dictionary.json", 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)

    max_sequence_length = 130
    question1_word_sequence = convert_text_to_index_array(question_1, dictionary)
    question1_word_sequence = [question1_word_sequence]
    question2_word_sequence = convert_text_to_index_array(question_2, dictionary)
    question2_word_sequence = [question2_word_sequence]

    question1_data = pad_sequences(question1_word_sequence, maxlen=max_sequence_length)
    question2_data = pad_sequences(question2_word_sequence, maxlen=max_sequence_length)

    model = load_siamese_model()

    prediction = model.predict([question1_data, question2_data])

    return prediction
# ============================ END SIAMESE MODEL ============================

def find_similar_question(sent):
    with open("..\\data\\questions.json", encoding="utf-8", errors="ignore") as file:
        data_questions = json.load(file)
        scores = []

        for i in range(10):
            question = data_questions[i]["question"]
            score = find_if_duplicate_questions(sent, question)
            scores.append(score)

        minimum = 9999

        for i in range(len(scores)):
            if scores[i] < minimum:
                minimum = i

        question = data_questions[minimum]["question"]
        answer = data_questions[minimum]["answer"]

        return question, answer


class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionParser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = "Hello, I am Mr. Healthcare. How can I help you?"
        result_classify = self.classifier.classify(sent)

        if not result_classify:
            question, stage2_answer = find_similar_question(sent)
            return f"I don't know if I fully understand the question, but here's what I found: \n {stage2_answer}"

        result_sql = self.parser.parser_main(result_classify)

        final_answers = self.searcher.search_main(result_sql)

        if not final_answers:
            # Perform Stage 2: Using datasets by checking question similarity in a loop and extracting answer from the least score
            question, stage2_answer = find_similar_question(sent)
            return f"I don't know if I fully understand the question, but here's what I found: \n {stage2_answer}"
        else:
            return '\n'.join(final_answers)


# ============================ GUI ============================
class ChatInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.sent_label = None
        self.master = master

        # sets default bg for top level windows
        self.tl_bg = "#EEEEEE"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"
        self.font = "Verdana 10"

        # Menu bar
        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)

        # File
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
        # file.add_command(label="Save Chat Log", command=self.save_chat)
        file.add_command(label="Clear Chat", command=self.clear_chat)
        #  file.add_separator()
        file.add_command(label="Exit", command=self.chat_exit)

        # Options
        options = Menu(menu, tearoff=0)
        menu.add_cascade(label="Options", menu=options)

        # username

        # font
        font = Menu(options, tearoff=0)
        options.add_cascade(label="Font", menu=font)
        font.add_command(label="Default", command=self.font_change_default)
        font.add_command(label="Times", command=self.font_change_times)
        font.add_command(label="System", command=self.font_change_system)
        font.add_command(label="Helvetica", command=self.font_change_helvetica)
        font.add_command(label="Fixedsys", command=self.font_change_fixedsys)

        # color theme
        color_theme = Menu(options, tearoff=0)
        options.add_cascade(label="Color Theme", menu=color_theme)
        color_theme.add_command(
            label="Default", command=self.color_theme_default)
        color_theme.add_command(label="Grey", command=self.color_theme_grey)
        color_theme.add_command(
            label="Blue", command=self.color_theme_dark_blue)
        color_theme.add_command(
            label="Torque", command=self.color_theme_turquoise)
        color_theme.add_command(
            label="Hacker", command=self.color_theme_hacker)

        help_option = Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=help_option)
        # help_option.add_command(label="Features", command=self.features_msg)
        help_option.add_command(label="About Aarogya-Bot", command=self.msg)
        help_option.add_command(label="Developers", command=self.about)

        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)

        # scrollbar for text box
        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)

        # contains messages
        self.text_box = Text(self.text_frame, yscrollcommand=self.text_box_scrollbar.set, state=DISABLED,
                             bd=1, padx=6, pady=6, spacing3=8, wrap=WORD, font="Verdana 10", relief=GROOVE,
                             width=10, height=1)
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        # frame containing user entry field
        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # entry field
        self.entry_field = Entry(self.entry_frame, bd=1, justify=LEFT)
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=3)
        # self.users_message = self.entry_field.get()

        # frame containing send button and emoji button
        self.send_button_frame = Frame(self.master, bd=0)
        self.send_button_frame.pack(fill=BOTH)

        # send button
        self.send_button = Button(self.send_button_frame, text="Send", width=5, relief=GROOVE, bg='white',
                                  bd=1, command=lambda: self.send_message_insert(None), activebackground="#FFFFFF",
                                  activeforeground="#000000")
        self.send_button.pack(side=LEFT, ipady=8)
        self.master.bind("<Return>", self.send_message_insert)

        self.last_sent_label(date="No messages sent.")
        # t2 = threading.Thread(target=self.send_message_insert(, name='t1')
        # t2.start()

    @staticmethod
    def play_response(response):
        x = pyttsx3.init()
        voices = x.getProperty('voices')
        x.setProperty('voice', voices[2].id)
        # print(response)
        li = []
        if len(response) > 100:
            if response.find('--') == -1:
                b = response.split('--')
                # print(b)

        x.setProperty('rate', 120)
        # x.setProperty('volume',50)
        x.say(response)
        x.runAndWait()
        # print("Played Successfully......")

    def last_sent_label(self, date):

        try:
            self.sent_label.destroy()
        except AttributeError:
            pass

        self.sent_label = Label(
            self.entry_frame, font="Verdana 7", text=date, bg=self.tl_bg2, fg=self.tl_fg)
        self.sent_label.pack(side=LEFT, fill=X, padx=3)

    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.last_sent_label(date="No messages sent.")
        self.text_box.delete(1.0, END)
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    @staticmethod
    def chat_exit():
        exit()

    @staticmethod
    def msg():
        tkinter.messagebox.showinfo("Aarogya-Bot info")

    @staticmethod
    def about():
        tkinter.messagebox.showinfo("Developers")

    def send_message_insert(self, message):
        # strMsg = message
        self.text_box.tag_config('red', foreground='#ff0000')
        user_input = self.entry_field.get()
        print(user_input)
        pr0 = "Human : " + \
              time.strftime('%B %d, %Y' + ' at ' + '%I:%M %p') + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr0, "red")
        self.entry_field.delete(0, END)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        pr1 = user_input + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr1)
        self.entry_field.delete(0, END)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        text2 = handler.chat_main(user_input) + '\n '
        print(text2)
        str_msg2 = text2
        pr3 = "Aarogya Bot:  : " + \
              time.strftime('%B %d, %Y' + ' at ' + '%I:%M %p') + "\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, pr3, "red")
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, str_msg2)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        self.last_sent_label(
            str(time.strftime("Last message sent: " + '%B %d, %Y' + ' at ' + '%I:%M %p')))
        self.entry_field.delete(0, END)
        time.sleep(0)
        # ob=chat(user_input)
        # pr="PyBot : " + ob + "\n"
        # self.text_box.configure(state=NORMAL)

        # self.text_box.configure(state=DISABLED)
        # self.text_box.see(END)
        # self.last_sent_label(str(time.strftime( "Last message sent: " + '%B %d, %Y' + ' at ' + '%I:%M %p')))
        # self.entry_field.delete(0,END)
        # time.sleep(0)
        # t2 = threading.Thread(target=self.playResponse, args=(text2, ))
        # t2.start()
        # return ob

    def font_change_default(self):
        self.text_box.config(font="Verdana 10")
        self.entry_field.config(font="Verdana 10")
        self.font = "Verdana 10"

    def font_change_times(self):
        self.text_box.config(font="Times")
        self.entry_field.config(font="Times")
        self.font = "Times"

    def font_change_system(self):
        self.text_box.config(font="System")
        self.entry_field.config(font="System")
        self.font = "System"

    def font_change_helvetica(self):
        self.text_box.config(font="helvetica 10")
        self.entry_field.config(font="helvetica 10")
        self.font = "helvetica 10"

    def font_change_fixedsys(self):
        self.text_box.config(font="fixedsys")
        self.entry_field.config(font="fixedsys")
        self.font = "fixedsys"

    def color_theme_default(self):
        self.master.config(bg="#EEEEEE")
        self.text_frame.config(bg="#EEEEEE")
        self.entry_frame.config(bg="#EEEEEE")
        self.text_box.config(bg="#FFFFFF", fg="#000000")
        self.entry_field.config(
            bg="#FFFFFF", fg="#000000", insertbackground="#000000")
        self.send_button_frame.config(bg="#EEEEEE")
        self.send_button.config(
            bg="#FFFFFF", fg="#000000", activebackground="#FFFFFF", activeforeground="#000000")
        # self.emoji_button.config(bg="#FFFFFF", fg="#000000", activebackground="#FFFFFF", activeforeground="#000000")
        self.sent_label.config(bg="#EEEEEE", fg="#000000")

        self.tl_bg = "#FFFFFF"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"

    # Dark
    def color_theme_dark(self):
        self.master.config(bg="#2a2b2d")
        self.text_frame.config(bg="#2a2b2d")
        self.text_box.config(bg="#212121", fg="#FFFFFF")
        self.entry_frame.config(bg="#2a2b2d")
        self.entry_field.config(
            bg="#212121", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#2a2b2d")
        self.send_button.config(
            bg="#212121", fg="#FFFFFF", activebackground="#212121", activeforeground="#FFFFFF")
        # self.emoji_button.config(bg="#212121", fg="#FFFFFF", activebackground="#212121", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#2a2b2d", fg="#FFFFFF")

        self.tl_bg = "#212121"
        self.tl_bg2 = "#2a2b2d"
        self.tl_fg = "#FFFFFF"

    # Grey
    def color_theme_grey(self):
        self.master.config(bg="#444444")
        self.text_frame.config(bg="#444444")
        self.text_box.config(bg="#4f4f4f", fg="#ffffff")
        self.entry_frame.config(bg="#444444")
        self.entry_field.config(
            bg="#4f4f4f", fg="#ffffff", insertbackground="#ffffff")
        self.send_button_frame.config(bg="#444444")
        self.send_button.config(
            bg="#4f4f4f", fg="#ffffff", activebackground="#4f4f4f", activeforeground="#ffffff")
        # self.emoji_button.config(bg="#4f4f4f", fg="#ffffff", activebackground="#4f4f4f", activeforeground="#ffffff")
        self.sent_label.config(bg="#444444", fg="#ffffff")

        self.tl_bg = "#4f4f4f"
        self.tl_bg2 = "#444444"
        self.tl_fg = "#ffffff"

        self.master.config(bg="#003333")
        self.text_frame.config(bg="#003333")
        self.text_box.config(bg="#669999", fg="#FFFFFF")
        self.entry_frame.config(bg="#003333")
        self.entry_field.config(
            bg="#669999", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#003333")
        self.send_button.config(
            bg="#669999", fg="#FFFFFF", activebackground="#669999", activeforeground="#FFFFFF")
        # self.emoji_button.config(bg="#669999", fg="#FFFFFF", activebackground="#669999", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#003333", fg="#FFFFFF")

        self.tl_bg = "#669999"
        self.tl_bg2 = "#003333"
        self.tl_fg = "#FFFFFF"

    # Blue
    def color_theme_dark_blue(self):
        self.master.config(bg="#263b54")
        self.text_frame.config(bg="#263b54")
        self.text_box.config(bg="#1c2e44", fg="#FFFFFF")
        self.entry_frame.config(bg="#263b54")
        self.entry_field.config(
            bg="#1c2e44", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#263b54")
        self.send_button.config(
            bg="#1c2e44", fg="#FFFFFF", activebackground="#1c2e44", activeforeground="#FFFFFF")
        # self.emoji_button.config(bg="#1c2e44", fg="#FFFFFF", activebackground="#1c2e44", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#263b54", fg="#FFFFFF")

        self.tl_bg = "#1c2e44"
        self.tl_bg2 = "#263b54"
        self.tl_fg = "#FFFFFF"

    # Torque

    def color_theme_turquoise(self):
        self.master.config(bg="#003333")
        self.text_frame.config(bg="#003333")
        self.text_box.config(bg="#669999", fg="#FFFFFF")
        self.entry_frame.config(bg="#003333")
        self.entry_field.config(
            bg="#669999", fg="#FFFFFF", insertbackground="#FFFFFF")
        self.send_button_frame.config(bg="#003333")
        self.send_button.config(
            bg="#669999", fg="#FFFFFF", activebackground="#669999", activeforeground="#FFFFFF")
        # self.emoji_button.config(bg="#669999", fg="#FFFFFF", activebackground="#669999", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#003333", fg="#FFFFFF")

        self.tl_bg = "#669999"
        self.tl_bg2 = "#003333"
        self.tl_fg = "#FFFFFF"

    # Hacker
    def color_theme_hacker(self):
        self.master.configure()
        self.master.config(bg="#0F0F0F")
        self.text_frame.config(bg="#0F0F0F")
        self.entry_frame.config(bg="#0F0F0F")
        self.text_box.config(bg="#0F0F0F", fg="#33FF33")
        self.entry_field.config(
            bg="#0F0F0F", fg="#33FF33", insertbackground="#33FF33")
        self.send_button_frame.config(bg="#0F0F0F")
        self.send_button.config(
            bg="#0F0F0F", fg="#FFFFFF", activebackground="#0F0F0F", activeforeground="#FFFFFF")
        # self.emoji_button.config(bg="#0F0F0F", fg="#FFFFFF", activebackground="#0F0F0F", activeforeground="#FFFFFF")
        self.sent_label.config(bg="#0F0F0F", fg="#33FF33")

        self.tl_bg = "#0F0F0F"
        self.tl_bg2 = "#0F0F0F"
        self.tl_fg = "#33FF33"

    # Default font and color theme

    def default_format(self):
        self.font_change_default()
        self.color_theme_default()


if __name__ == '__main__':
    root = Tk()
    a = ChatInterface(root)
    root.geometry(window_size)
    root.title("Aarogya-Bot")
    handler = ChatBotGraph()
    root.mainloop()
