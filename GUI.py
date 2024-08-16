from tkinter import *
import joblib
import pandas as pd
import tkinter.font as font

def show_entry_fields():
    # Retrieve values from the option menus and entry fields
    text = clicked.get()
    p1 = 1 if text == "Male" else 0
    
    # Access the values from Entry widgets
    p2 = float(e2.get())
    text = clicked1.get()
    p3 = 1 if text == "Central" else 0
    p4 = float(e4.get())
    text = clicked6.get()
    p5 = 1 if text == "Central" else 0
    text = clicked2.get()
    p6 = 2 if text == "Science" else 1 if text == "Commerce" else 0
    p7 = float(e7.get())
    text = clicked3.get()
    p8 = 2 if text == "Sci&Tech" else 1 if text == "Comm&Mgmt" else 0
    text = clicked4.get()
    p9 = 1 if text == "Yes" else 0
    p10 = float(e10.get())
    text = clicked5.get()
    p11 = 1 if text == "Mkt&HR" else 0
    p12 = float(e12.get())

    # Load the model and make predictions
    model = joblib.load('model_campus_placement')
    new_data = pd.DataFrame({
        'gender': [p1],
        'ssc_p': [p2],
        'ssc_b': [p3],
        'hsc_p': [p4],
        'hsc_b': [p5],
        'hsc_s': [p6],
        'degree_p': [p7],
        'degree_t': [p8],
        'workex': [p9],
        'etest_p': [p10],
        'specialisation': [p11],
        'mba_p': [p12],   
    })
    
    result = model.predict(new_data)
    result1 = model.predict_proba(new_data)

    result_frame = Frame(master, bg='lightgray', pady=10)
    result_frame.grid(row=30, column=0, columnspan=2, sticky=W+E)

    if result[0] == 0:
        Label(result_frame, text="Can't be Placed", font=("Arial", 16), bg='lightgray', fg='red').pack()
    else:
        Label(result_frame, text="Student Will be Placed With Probability of", font=("Arial", 16, 'bold'), bg='lightgray').pack()
        Label(result_frame, text=f"{round(result1[0][1], 2) * 100}%", font=("Arial", 20, 'bold'), bg='lightgray').pack()
        Label(result_frame, text="Percent", font=("Arial", 16), bg='lightgray').pack()

master = Tk()
master.title("Campus Placement Prediction System")

# Make the window full screen
master.attributes('-fullscreen', True)

# Optionally, hide the window decorations
master.overrideredirect(True)

master.configure(bg='lightblue')

header_font = font.Font(family='Arial', size=20, weight='bold')
label = Label(master, text="Campus Placement Prediction System", bg="darkblue", fg="white", font=header_font, pady=10)
label.grid(row=0, column=0, columnspan=2, sticky=W+E)

# Initialize StringVar for OptionMenu widgets
clicked = StringVar()
clicked1 = StringVar()
clicked2 = StringVar()
clicked3 = StringVar()
clicked4 = StringVar()
clicked5 = StringVar()
clicked6 = StringVar()

# Default values for StringVar
clicked.set("Male")
clicked1.set("Central")
clicked2.set("Science")
clicked3.set("Sci&Tech")
clicked4.set("Yes")
clicked5.set("Mkt&HR")
clicked6.set("Central")

# Create Entry widgets and store their references
e2 = Entry(master)
e4 = Entry(master)
e7 = Entry(master)
e10 = Entry(master)
e12 = Entry(master)

fields = [
    ("Gender", clicked, ["Male", "Female"]),
    ("Secondary Education percentage - 10th Grade", e2),
    ("Board of Education", clicked1, ["Central", "Others"]),
    ("Higher Secondary Education percentage - 12th Grade", e4),
    ("Board of Education", clicked6, ["Central", "Others"]),
    ("Specialization in Higher Secondary Education", clicked2, ["Science", "Commerce", "Arts"]),
    ("Degree Percentage", e7),
    ("Under Graduation(Degree type) - Field of degree education", clicked3, ["Sci&Tech", "Comm&Mgmt", "Others"]),
    ("Work Experience", clicked4, ["Yes", "No"]),
    ("Enter test percentage", e10),
    ("Branch specialization", clicked5, ["Mkt&HR", "Mkt&Fin"]),
    ("MBA percentage", e12)
]

row = 1
for label_text, widget, *options in fields:
    Label(master, text=label_text, font=("Arial", 14), bg='lightblue').grid(row=row, column=0, sticky=W, padx=20, pady=5)
    if options:
        option_menu = OptionMenu(master, widget, *options[0])
        option_menu.configure(width=30)
        option_menu.grid(row=row, column=1, padx=20, pady=5)
    else:
        widget.configure(width=30)
        widget.grid(row=row, column=1, padx=20, pady=5)
    row += 1

button_font = font.Font(family='Helvetica', size=16, weight='bold')
Button(master, text='Predict', height=2, width=15, activebackground='#00ff00', font=button_font, bg='black', fg='white', command=show_entry_fields).grid(row=row, column=0, columnspan=2, pady=20)

master.mainloop()
