import tkinter as tk

def build_gui():
    root = tk.Tk()
    root.title("GUI")
    root.geometry("500x500")
    root.resizable(False, False)
    build_main_frame(root)
    root.mainloop()
    return root

def build_main_frame(root):
    search_frame = tk.Frame(root, width=500, height=500, bg="#00acee")
    search_frame.pack()
    return search_frame


build_gui()