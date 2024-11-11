import tkinter as tk
from functools import partial
import threading
import socket
from tkinter import messagebox

# Kích thước bàn cờ
Ox = 15
Oy = 15

class Window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Caro Game")
        self.Buts = {}
        self.memory = []
        self.game_over = False
        self.Threading_socket = Threading_socket(self)

    def showFrame(self):
        frame1 = tk.Frame(self)
        frame1.pack()
        frame2 = tk.Frame(self)
        frame2.pack()

        # Nút Undo và New Game
        Undo = tk.Button(frame1, text="Undo", width=10,
                        command=partial(self.Undo, synchronized=True))
        Undo.grid(row=0, column=0, padx=30)

        newGameBtn = tk.Button(frame1, text="New Game", width=10,
                             command=self.newGame)
        newGameBtn.grid(row=0, column=1, padx=30)

        # Các nút kết nối mạng
        tk.Label(frame1, text="IP", pady=4).grid(row=0, column=2)
        inputIp = tk.Entry(frame1, width=20)
        inputIp.grid(row=0, column=3, padx=5)
        connectBT = tk.Button(frame1, text="Connect", width=10,
                            command=lambda: self.Threading_socket.clientAction(inputIp.get()))
        connectBT.grid(row=0, column=4, padx=3)

        makeHostBT = tk.Button(frame1, text="MakeHost", width=10,
                              command=lambda: self.Threading_socket.serverAction())
        makeHostBT.grid(row=0, column=5, padx=30)

        # Tạo bàn cờ
        for x in range(Ox):
            for y in range(Oy):
                self.Buts[x, y] = tk.Button(frame2, font=('arial', 15, 'bold'), height=1, width=2,
                                          borderwidth=2, command=partial(self.handleButton, x=x, y=y))
                self.Buts[x, y].grid(row=x, column=y)

    def handleButton(self, x, y):
        if not self.game_over and self.Buts[x, y]['text'] == "":
            if self.memory.count([x, y]) == 0:
                self.memory.append([x, y])
            if len(self.memory) % 2 == 1:
                self.Buts[x, y]['text'] = 'O'
                self.Threading_socket.sendData("{}|{}|{}|".format("hit", x, y))
                if self.checkWin(x, y, "O"):
                    self.game_over = True
                    self.notification("Game Over", "Player O Wins!")
            else:
                self.Buts[x, y]['text'] = 'X'
                self.Threading_socket.sendData("{}|{}|{}|".format("hit", x, y))
                if self.checkWin(x, y, "X"):
                    self.game_over = True
                    self.notification("Game Over", "Player X Wins!")

    def checkWin(self, x, y, XO):
        # Kiểm tra hàng ngang
        count = 1
        # Kiểm tra bên phải
        for i in range(1, 5):
            if y + i < Oy and self.Buts[x, y + i]['text'] == XO:
                count += 1
            else:
                break
        # Kiểm tra bên trái
        for i in range(1, 5):
            if y - i >= 0 and self.Buts[x, y - i]['text'] == XO:
                count += 1
            else:
                break
        if count >= 5:
            return True

        # Kiểm tra hàng dọc
        count = 1
        # Kiểm tra phía dưới
        for i in range(1, 5):
            if x + i < Ox and self.Buts[x + i, y]['text'] == XO:
                count += 1
            else:
                break
        # Kiểm tra phía trên
        for i in range(1, 5):
            if x - i >= 0 and self.Buts[x - i, y]['text'] == XO:
                count += 1
            else:
                break
        if count >= 5:
            return True

        # Kiểm tra đường chéo chính
        count = 1
        for i in range(1, 5):
            if x + i < Ox and y + i < Oy and self.Buts[x + i, y + i]['text'] == XO:
                count += 1
            else:
                break
        for i in range(1, 5):
            if x - i >= 0 and y - i >= 0 and self.Buts[x - i, y - i]['text'] == XO:
                count += 1
            else:
                break
        if count >= 5:
            return True

        # Kiểm tra đường chéo phụ
        count = 1
        for i in range(1, 5):
            if x + i < Ox and y - i >= 0 and self.Buts[x + i, y - i]['text'] == XO:
                count += 1
            else:
                break
        for i in range(1, 5):
            if x - i >= 0 and y + i < Oy and self.Buts[x - i, y + i]['text'] == XO:
                count += 1
            else:
                break
        if count >= 5:
            return True

        return False

    def notification(self, title, msg):
        messagebox.showinfo(title, msg)

    def newGame(self):
        self.game_over = False
        for x in range(Ox):
            for y in range(Oy):
                self.Buts[x, y]['text'] = ""
        self.memory = []

    def Undo(self, synchronized):
        if len(self.memory) > 0:
            x = self.memory[-1][0]
            y = self.memory[-1][1]
            self.Buts[x, y]['text'] = ""
            self.memory.pop()
            if synchronized:
                self.Threading_socket.sendData("{}|".format("Undo"))

class Threading_socket():
    def __init__(self, gui):
        self.dataReceive = ""
        self.conn = None
        self.gui = gui
        self.name = ""

    def clientAction(self, inputIP):
        self.name = "client"
        HOST = inputIP
        PORT = 8000
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((HOST, PORT))
        self.gui.notification("Connected to", HOST)
        threading.Thread(target=self.client).start()

    def client(self):
        while True:
            try:
                self.dataReceive = self.conn.recv(1024).decode()
                if self.dataReceive:
                    parts = self.dataReceive.split("|")
                    if len(parts) >= 2:
                        friend = parts[0]
                        action = parts[1]
                        if action == "hit" and friend == "server" and len(parts) >= 4:
                            x = int(parts[2])
                            y = int(parts[3])
                            self.gui.handleButton(x, y)
                        elif action == "Undo" and friend == "server":
                            self.gui.Undo(False)
                self.dataReceive = ""
            except:
                break

    def serverAction(self):
        self.name = "server"
        HOST = socket.gethostbyname(socket.gethostname())
        PORT = 8000
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(1)
        self.gui.notification("Your IP", HOST)
        self.conn, addr = s.accept()
        threading.Thread(target=self.server, args=(addr, s)).start()

    def server(self, addr, s):
        try:
            while True:
                self.dataReceive = self.conn.recv(1024).decode()
                if self.dataReceive:
                    parts = self.dataReceive.split("|")
                    if len(parts) >= 2:
                        friend = parts[0]
                        action = parts[1]
                        if action == "hit" and friend == "client" and len(parts) >= 4:
                            x = int(parts[2])
                            y = int(parts[3])
                            self.gui.handleButton(x, y)
                        elif action == "Undo" and friend == "client":
                            self.gui.Undo(False)
                self.dataReceive = ""
        finally:
            s.close()

    def sendData(self, data):
        if self.conn:
            self.conn.sendall(f"{self.name}|{data}".encode())

if __name__ == "__main__":
    window = Window()
    window.showFrame()
    window.mainloop()
