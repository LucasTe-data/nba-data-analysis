# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:39:40 2020

@author: Lucas
"""


import tkinter as tk
import pickle
import numpy as np

class Basketball_Application(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master = master
        self.create_widgets()
    
    
    def create_widgets(self):
        color = {'text' : '#ffffff',
            'button': '#474749',
            'entry' : '#474749',
            'background' : '#2a2a2e'}
        
        self.titel = tk.Label(text = 'Which Basketball Position is best for you?', font = (None, 12), background = color['background'], fg = color['text'], bd = 10)
        self.titel.grid(row = 0, columnspan = 2)
        
        self.weight_label = tk.Label(text = 'What is your weight? (kg)', background = color['background'], fg = color['text'], bd = 5)
        self.weight_label.grid(row = 1, column = 0)
        self.weight_entry = tk.Entry(background = color['entry'], fg = color['text'])
        self.weight_entry.grid(row = 1, column = 1, padx=(0,10))
        
        self.height_label = tk.Label(text = 'What is your height? (cm)', background = color['background'], fg = color['text'], bd = 5)
        self.height_label.grid(row = 2, column = 0, sticky = 'W')
        self.height_entry = tk.Entry(background = color['entry'], fg = color['text'])
        self.height_entry.grid(row = 2, column = 1, padx=(0,10))        
        
        self.result_label = tk.Label(text = 'You should play as ---', background = color['background'], fg = color['text'], bd = 5)
        self.result_label.grid(row = 3, columnspan = 2)
        
        self.predict_button = tk.Button(command = lambda: self.predict(weight = self.weight_entry.get(), height = self.height_entry.get()), text = 'Predict', background = color['button'], fg = color['text'], relief = 'groove')
        self.predict_button.grid(row = 4, columnspan = 2, sticky = 'W', padx = (60,0), pady = (0,15))
        
        self.clear_button = tk.Button(text = 'Clear', command = self.clear_text, background = color['button'], fg = color['text'], relief = 'groove')
        self.clear_button.grid(row = 4, columnspan = 2, pady = (0,15))
        
        self.quit_all = tk.Button(text = 'Quit', command =self.master.destroy, background = color['button'], fg = color['text'], relief = 'groove')
        self.quit_all.grid(row = 4, columnspan = 2, sticky = 'E', padx = (0,75), pady = (0,15))
        
    def predict(self, weight, height):
        try:
            weight = float(weight)
            height = float(height)
            
            variables = [[weight, height]]            
            
            filename = 'position_model.pkl'
            loaded_model = pickle.load(open(filename, 'rb'))
            classes_list = loaded_model.classes_
            position = loaded_model.predict(variables)
            position_proba = loaded_model.predict_proba(variables)[0]
            position_index = np.where(classes_list == position)        
            
            self.result_label.config(text = F'You should play as {position[0]} ({round(position_proba[position_index][0]*100,2)}%)')
        except ValueError:
            tk.messagebox.showinfo("Error","At least one variable is not a number")
        
        
    def clear_text(self):
        self.weight_entry.delete(0, 'end')
        self.height_entry.delete(0, 'end')
        
        self.result_label.configure(text = 'You should play as ---')
        
        
        
top = tk.Tk()
top.title('Basketball Position')
top.configure(background = '#2a2a2e')
app = Basketball_Application(master = top)
app.mainloop()