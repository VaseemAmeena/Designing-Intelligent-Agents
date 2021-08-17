'''
# If it does not work on your environment, please run this code (with the dataset) on Jupyter notebook from the virtual comp sci
# desktop. You can run it by copy pasting this code on a new python file and uploading the 'dataset.JSON' file
# on Jupyter notebook.
# I have tested this code on multiple environments, and it should work since nltk and keras packages
# are pre installed (as confirmed through email).
'''
# latest

# importing Natural language toolkit package for text processing
import nltk as lang_toolkit

lang_toolkit.download('punkt')
lang_toolkit.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

# importing packages for model creation
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model

# importing other packages
import pickle
import json
import random
import numpy as np




# function to call the model easily
def getmodel():
    tokens = []
    docs = []
    IDs = []
    filterOut = ['?', '!']
    
 #reading from dataset to implement text pre processing
    data_file = open('dataset.json').read()
    dataset = json.loads(data_file)
    # preprocessing text data before it can be used by algorithm
    for user_input in dataset['dataset']:
        for pt in user_input['patterns']:
            #tokenization of input
            w_tok = lang_toolkit.word_tokenize(pt)
            tokens.extend(w_tok)
            #adding document
            docs.append((w_tok, user_input['ID']))  
            if user_input['ID'] not in IDs:
                #put ID in list
                IDs.append(user_input['ID'])  

    #apply lower casing and lemmatisation
    tokens = [lemma.lemmatize(w_tok.lower()) for w_tok in tokens if w_tok not in filterOut]
    
    # sorting
    tokens = sorted(list(set(tokens)))
    IDs = sorted(list(set(IDs)))

    # storing the Python objects in pickle files to be used while predicting
    getTokens = open('tokens.pkl', 'wb')
    pickle.dump(tokens, getTokens)
    getIDs = open('IDs.pkl', 'wb')
    pickle.dump(IDs, getIDs)

    # initialising data to train model
    data_train = []

    base_response = len(IDs) * [0]
    # bag of words (from sentences)
    for doc in docs:
        bow = []
        # choice of tokens (tokenized words) for the input(s)
        inputWords = doc[0]
        # lemmatization by grouping together the inflected forms of a word into their lemma form
        inputWords = [lemma.lemmatize(w.lower())for w in inputWords]


        # if current input has match then update with '1'
        for w_tok in tokens:
            bow.append(1) if w_tok in inputWords else bow.append(0)

        # response would be 1 for current ID (of input) while 0 for other IDs
        list_response = list(base_response)
        list_response[IDs.index(doc[1])] = 1

        data_train.append([bow, list_response])
    # create np.array with randomised features
    random.shuffle(data_train)
    data_train = np.array(data_train)

    # creating data for training and testing with a for input(s) and b for IDs
    a_Data = list(data_train[:, 0])
    b_Data = list(data_train[:, 1])

    # building the sequential classification model
    classif_mod = Sequential()
   
    # parameters chosen after testing and tuning multiple times until a good accuracy was reached
    # building the first layer with 130 nodes
    k = len(a_Data[0])
    classif_mod.add(Dense(130, batch_input_shape=(None, k), activation='relu'))
    classif_mod.add(Dropout(0.5))

    # building the second layer with 70 nodes
    classif_mod.add(Dense(70, activation='relu'))
    classif_mod.add(Dropout(0.5))

    # third layer contains the same number of nodes as sections in dataset

    b_length = len(b_Data[0])
    classif_mod.add(Dense(b_length, activation='softmax'))

    optim = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    classif_mod.compile(metrics=['accuracy'], optimizer=optim, loss='categorical_crossentropy')
    

    # Epoch size kept at 215
    runs = classif_mod.fit(np.array(a_Data), np.array(b_Data), verbose = 1, batch_size=5, epochs=215)
    classif_mod.save('final.h5', runs)


# adds name to dataset
def addToDataset(data, datafile="dataset.json"):
    with open(datafile, "w")as wordfile:
        json.dump(data, wordfile, indent = 4)


# loads model, dataset, and pickle files
def modelRun():
    getmodel()
    classif_mod = load_model('final.h5')


    datas = 'dataset.json'
    IDp = 'IDs.pkl'
    tpkl = 'tokens.pkl'
    dataset = json.loads(open(datas).read())
    IDs = pickle.load(open(IDp, 'rb'))
    tokens = pickle.load(open(tpkl, 'rb'))      
    list1 = []
    list1.append(classif_mod)
    list1.append(dataset)
    list1.append(tokens)
    list1.append(IDs)
    return list1


def token_lemm(sent):
    # input tokenisation
    w_sent = lang_toolkit.word_tokenize(sent)
    # lemmatization of words
    w_sent = [lemma.lemmatize(w.lower()) for w in w_sent]
    return w_sent


def get_ID(sent, classif_mod):
    info = bagThis(sent, tokens, deets=False)
    output = classif_mod.predict(np.array([info]))[0]
    error_limit = 0.25  # eliminate predictions within an error threshold
    finalValues = [[d, randval] for d, randval in enumerate(output) if randval > error_limit]
    #sorting according to highest prob
    finalValues.sort(key=lambda x: x[1] , reverse = True)
    ID_vals = []
    for randval in finalValues:
        ID_vals.append({"user_input": IDs[randval[0]], "prob": str(randval[1])})
    return ID_vals


# return array for bag of words with 1 for each match between a word in the sent and word in the bag
def bagThis(sent, tokens, deets=True):
    # input tokenization
    w_sent = token_lemm(sent)
    # creating bag of words

   

    bow = [0] * len(tokens)
    for wx in w_sent:
        for d, w_tok in enumerate(tokens):
            if w_tok == wx:
                # if word match in vocab then 1
                bow[d] = 1
                if deets:
                    print("the contents of the bag are: %s" % w_tok)
    return (np.array(bow))


def get_output(data_vals, dataset_file):
    # matches closest value of user input from data_vals to the dataset
    try:
        ID = data_vals[0]['user_input']
    except:
        # print error message from error ID if question is not in the dataset
        ID = 'error'
    dataset_vals = dataset_file['dataset']
    for d in dataset_vals:
        if (d['ID'] == ID):
            # choose a random response if there are multiple options (small talk)
            final_val = random.choice(d['responses'])
            break
    return final_val


# predicts closest ID to the message from the model
def bot_reply(msg):
    data_vals = get_ID(msg, classif_mod)
    output = get_output(data_vals, dataset)
    return output


def send(msg):
    if msg != '':
        print('\n' + "You: " + msg + '\n')

        output = bot_reply(msg)
        print("Bot: " + output + '\n\n')

# adding name element to JSON file
def thisisit():
    with open("dataset.json", 'r+') as jsonfile:
        data=json.load(jsonfile)
        temp = data["dataset"]
        for element in temp:
            if element['ID'] == "Names":
                temp.remove(element)
        jsonfile.seek(0)
        json.dump(data, jsonfile, indent=4)
        jsonfile.truncate()

    with open("dataset.json") as jsonfile:
        data = json.load(jsonfile)
        temp = data["dataset"]

        name = input("What is your name? ")
        y = {"ID": "Names",
             "patterns": ["my name is {}".format(name), "call me {}".format(name), "I am {} ".format(name),
                          "what is my name?"],
             "responses": ["hello,{}. Please ask me a question".format(name), "hi,{},what would you like to know?".format(name)],
             "context": ["search_name"]
             }
        temp.append(y)
    addToDataset(data)
    return name





n = thisisit()
getemfile = modelRun()
classif_mod = getemfile[0]
dataset = getemfile[1]
tokens = getemfile[2]
IDs = getemfile[3]
while True:
    entrybox = input("\n Please ask me a question and I will attempt to answer it from our dataset of natural questions.\n "
                     "You could also choose to say hi, bye, thanks, or ask for your name to see if I am able to remember it\n"
                     " Enter message here: ")
    namequestion = ["my name is", "call me", "I am"]
    contains = any(namequestion in entrybox for namequestion in namequestion)

    if contains == True:
        word_list = entrybox.split()
        possiblename = word_list[-1]
        print(possiblename)
        with open("dataset.json", 'r+') as jsonfile:
            data = json.load(jsonfile)
            temp = data["dataset"]
            for element in temp:
                if element['ID'] == "Names":
                    list_of_patterns = element.get('patterns')
                    for pt in list_of_patterns:
                        if possiblename in pt:
                            print("")  # if user inputs the same name as before, nothing wil change in dataset
                        else:
                            name = possiblename  # if user changes name, dataset is automatically updated and old name is replaced with new name
                            element['patterns'] = ["my name is {}".format(name), "call me {}".format(name),
                                                   "I am {} ".format(name), "what is my name?"]
                            element['responses'] = ["hello,{}. Please ask me a question".format(name),
                                                    "hi,{},what would you like to know?".format(name)]
                            jsonfile.seek(0)
                            json.dump(data, jsonfile, indent=4)
                            jsonfile.truncate()
                            getemfile = modelRun()
                            classif_mod = getemfile[0]
                            dataset = getemfile[1]
                            tokens = getemfile[2]
                            IDs = getemfile[3]
                            break

    if entrybox == "quit":
        break
    send(entrybox)