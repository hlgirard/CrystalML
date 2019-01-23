import tkinter as tk
from tkinter.messagebox import askquestion
from PIL import Image,ImageTk
import os
from functools import partial
import pickle


class ImageClassifier(tk.Frame):

    def __init__(self, parent, directory, categories, *args, **kwargs):

        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.root = parent
        self.root.wm_title("Manual Image labelling")

        # Dimensions
        self.winwidth = 1000
        self.imwidth = self.winwidth - 10
        self.imheight = int(self.imwidth // 1.5)

        #  Directory containing the raw images and saved dictionary
        self.folder = directory
        self.savepath = self.folder + "/labelled.pkl"

        # Categories for the labelling task
        self.categories = categories
        # Add default categories
        self.categories.append('Remove')

        # Initialize data
        self.initialize_data()

        # Make a frame for global control buttons
        self.frame0 = tk.Frame(self.root, width=self.winwidth, height=10, bd=2)
        self.frame0.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Make a frame to display the image
        self.frame1 = tk.Frame(self.root, width=self.winwidth, height=self.imheight+10, bd=2)
        self.frame1.pack(side=tk.TOP)

        # Create a canvas for the image
        self.cv1 = tk.Canvas(self.frame1, width=self.imwidth, height=self.imheight, background="white", bd=1, relief=tk.RAISED)
        self.cv1.pack(in_=self.frame1)

        # Make a frame to display the labelling buttons
        self.frame2 = tk.Frame(self.root, width=1000, height=10, bd=2)
        self.frame2.pack(side = tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create the global buttons
        tk.Button(self.root, text='Exit', height=2, width=8, command =self.quit).pack(in_=self.frame0, side = tk.RIGHT)
        tk.Button(self.root, text='Reset', height=2, width=8, command =self.reset_session).pack(in_=self.frame0, side = tk.RIGHT)
        tk.Button(self.root, text='DELETE ALL', height=2, width=10, command =self.delete_saved_data).pack(in_=self.frame0, side = tk.RIGHT)

        tk.Button(self.root, text='Save', height=2, width=8, command =self.save).pack(in_=self.frame0, side = tk.LEFT)
        tk.Button(self.root, text='Previous', height=2, width=8, command =self.previous_image).pack(in_=self.frame0, side = tk.LEFT)

        # Create a button for each of the categories
        for category in self.categories:
            tk.Button(self.root, text=category, height=2, width=8, command = partial(self.classify, category)).pack(in_=self.frame2, fill = tk.X, expand = True, side = tk.LEFT)

        self.next_image()

    def initialize_data(self):
        # Initialize dictionary
        if os.path.isfile(self.savepath):
            self.labeled = self.load_dict(self.savepath)
            print("Loaded existing dictionary from disk")
            print(self.labeled)
        else:
            self.labeled = {}
            print("No dictionary found, initializing a new one")

        # Build list of images to classify
        self.image_list = []
        for d in os.listdir(self.folder):
            if d not in self.labeled: 
                self.image_list.append(d)
        print("{} images ready to label".format(len(self.image_list)))

        # Initialize counter and get number of images   
        self.counter = 0
        self.max_count = len(self.image_list)-1

    def classify(self, category):
        self.labeled[self.image_list[self.counter]] = category
        print('Label {} selected for image {}'.format(category, self.image_list[self.counter]))
        self.counter += 1
        self.next_image()
    
    def previous_image(self):
        self.counter += -1
        self.next_image()

    def next_image(self):
        if self.counter > self.max_count:
            print("No more images")
        else:
            im = Image.open("{}{}".format(self.folder + '/', self.image_list[self.counter]))
            if (self.imwidth-im.size[0])<(self.imheight-im.size[1]):
                width = self.imwidth
                height = width*im.size[1]/im.size[0]
                self.display(height, width)
            else:
                height = self.imheight
                width = height*im.size[0]/im.size[1]
                self.display(height, width)

    def display(self, height, width):
        self.im = Image.open("{}{}".format(self.folder + '/', self.image_list[self.counter]))
        self.im.thumbnail((width, height), Image.ANTIALIAS)
        self.root.photo = ImageTk.PhotoImage(self.im)
        self.photo = ImageTk.PhotoImage(self.im)

        if self.counter == 0:
            self.cv1.create_image(0, 0, anchor = 'nw', image = self.photo)

        else:
            self.im.thumbnail((width, height), Image.ANTIALIAS)
            self.cv1.delete("all")
            self.cv1.create_image(0, 0, anchor = 'nw', image = self.photo)

    
    def save(self):
        self.dump_dict(self.labeled, self.savepath)
        print("Saved data to file")
    
    def load_dict(self, file):
        with open(file,"rb") as f:
            return pickle.load(f)
    
    def dump_dict(self, dict, file):
        with open(file, 'wb') as f:
            pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    def reset_session(self):
        result = askquestion('Are you sure?', 'Delete data since last save?', icon = 'warning')
        if result == 'yes':
            print("Resetting session since last save and reinitializing date")
            self.labeled = {}
            self.initialize_data()
            self.next_image()
        else:
            pass
    
    def delete_saved_data(self):
        result = askquestion('Are you sure?', 'Delete all saved and session data?', icon = 'warning')
        if result == 'yes':
            print("Deleting all saved progress and reinitializing data")
            if os.path.isfile(self.savepath):
                os.remove(self.savepath)
            self.initialize_data()
            self.next_image()
        else:
            pass

if __name__ == "__main__":
    root = tk.Tk() 
    rawDirectory = "data/raw"
    categories = ['Crystal', 'Clear', 'Aggregate']
    MyApp = ImageClassifier(root, rawDirectory, categories)
    tk.mainloop()