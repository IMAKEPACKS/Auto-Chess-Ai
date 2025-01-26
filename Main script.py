import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import pyautogui
import datetime
from PIL import Image, ImageDraw, ImageFont
import os
import tkinter as tk
from tkinter import ttk
import keyboard
from stockfish import Stockfish
import threading
import queue
import platform
import subprocess
import time
import random

class BoardRegionSelector:
    def __init__(self, parent):
        self.parent = parent
        self.root = tk.Toplevel(parent.root)
        self.root.title("Select Chess Board Region")
        self.root.attributes('-topmost', True)
        
       
        self.root.attributes('-alpha', 0.7)
        self.root.geometry("300x150")
        
    
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.selecting = False
        
        self.setup_gui()
        
    def setup_gui(self):
        instruction_label = ttk.Label(
            self.root,
            text="1. Press 'Start Selection'\n2. Click and drag to draw a box around your chess board\n3. Press 'Confirm' to save"
        )
        instruction_label.pack(pady=10)
        
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.select_button = ttk.Button(
            button_frame,
            text="Start Selection",
            command=self.start_selection
        )
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        self.confirm_button = ttk.Button(
            button_frame,
            text="Confirm",
            command=self.confirm_selection,
            state=tk.DISABLED
        )
        self.confirm_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_selection
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack(pady=10)
        
    def start_selection(self):
        self.root.iconify() 
        self.select_button.configure(state=tk.DISABLED)
        self.confirm_button.configure(state=tk.DISABLED)
        
      
        self.overlay = tk.Toplevel(self.root)
        self.overlay.attributes('-fullscreen', True, '-alpha', 0.3)
        self.overlay.configure(bg='black')
        
        self.canvas = tk.Canvas(
            self.overlay,
            highlightthickness=0,
            bg='black'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.selecting = True
        
    def on_mouse_drag(self, event):
        if self.selecting:
            self.current_x = event.x
            self.current_y = event.y
            self.canvas.delete("selection")
            self.canvas.create_rectangle(
                self.start_x, self.start_y,
                self.current_x, self.current_y,
                outline="red",
                width=2,
                tags="selection"
            )
            
    def on_mouse_up(self, event):
        self.selecting = False
        self.overlay.destroy()
        self.root.deiconify()  
        self.confirm_button.configure(state=tk.NORMAL)
        
      
        x1 = min(self.start_x, self.current_x)
        y1 = min(self.start_y, self.current_y)
        x2 = max(self.start_x, self.current_x)
        y2 = max(self.start_y, self.current_y)
        
        self.final_coords = {
            'top_left': (x1, y1),
            'bottom_right': (x2, y2)
        }
        
        self.status_label.configure(
            text=f"Selected region: ({x1}, {y1}) to ({x2}, {y2})"
        )
        
    def confirm_selection(self):
        if hasattr(self, 'final_coords'):
            self.parent.detector.board_coords = self.final_coords
            self.parent.log_message("Board region updated successfully")
            self.root.destroy()
        else:
            self.status_label.configure(text="Please make a selection first")
            
    def cancel_selection(self):
        self.root.destroy()

class ChessPieceNet(torch.nn.Module):
    def __init__(self, num_classes=13):
        super(ChessPieceNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 8 * 8, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x


class ChessScreenDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessPieceNet(num_classes=13)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.square_size = None
        self.board_origin = None
        
        self.board_coords = {
            'top_left': (300, 170),
            'bottom_right': (1140, 1010)
        }

        self.clock_coords = {
            'local': {'top_left': (999, 1020), 'bottom_right': (1135, 1057)},
            'enemy': {'top_left': (999, 121), 'bottom_right': (1136, 158)}
        }
        
        self.classes = [
            'wK', 'wQ', 'wR', 'wB', 'wN', 'wP',
            'bK', 'bQ', 'bR', 'bB', 'bN', 'bP',
            'eM'
        ]
        
        self.fen_mapping = {
            'wK': 'K', 'wQ': 'Q', 'wR': 'R', 'wB': 'B', 'wN': 'N', 'wP': 'P',
            'bK': 'k', 'bQ': 'q', 'bR': 'r', 'bB': 'b', 'bN': 'n', 'bP': 'p',
            'eM': '1'
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.debug_mode = False

    def find_chess_board(self):
        self.board_origin = (self.board_coords['top_left'][0], self.board_coords['top_left'][1])
        board_width = self.board_coords['bottom_right'][0] - self.board_coords['top_left'][0]
        self.square_size = board_width // 8
        return (
            self.board_coords['top_left'][0],
            self.board_coords['top_left'][1],
            self.board_coords['bottom_right'][0],
            self.board_coords['bottom_right'][1]
        )
    
    def get_square_center(self, square_notation, playing_as_white=True):
        """
        Convert chess square notation (e.g., 'e4') to screen coordinates.
        Handles both white and black perspective.
        
        Args:
            square_notation (str): Chess square notation (e.g., 'e4')
            playing_as_white (bool): Whether we're playing as white
            
        Returns:
            tuple: (x, y) screen coordinates for the center of the square
        """
        if not self.board_origin or not self.square_size:
            raise ValueError("Board dimensions not initialized")
        
        
        file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        
        file_idx = file_map[square_notation[0].lower()]
        rank_idx = 8 - int(square_notation[1])  
        
        if not playing_as_white:
            
            file_idx = 7 - file_idx
            rank_idx = 7 - rank_idx
        
        x = self.board_origin[0] + (file_idx * self.square_size) + (self.square_size // 2)
        y = self.board_origin[1] + (rank_idx * self.square_size) + (self.square_size // 2)
        
        return x, y
    
    def capture_screen(self):
        try:
            board_coords = self.find_chess_board()
            width = board_coords[2] - board_coords[0]
            height = board_coords[3] - board_coords[1]
            
            if width <= 0 or height <= 0:
                raise ValueError("Invalid board coordinates")
                
            screenshot = pyautogui.screenshot(region=(
                board_coords[0],
                board_coords[1],
                width,
                height
            ))
            return np.array(screenshot)
        except Exception as e:
            raise Exception(f"Screen capture failed: {str(e)}")

    def split_board_into_squares(self, board_image):
        height, width = board_image.shape[:2]
        square_height = height // 8
        square_width = width // 8
        squares = []
        positions = []

        for row in range(8):
            for col in range(8):
                y1 = row * square_height
                y2 = (row + 1) * square_height
                x1 = col * square_width
                x2 = (col + 1) * square_width
                
                square = board_image[y1:y2, x1:x2]
                squares.append(square)
                positions.append((x1, y1, x2, y2))

        return squares, positions
    
    def detect_piece(self, image_region):
        pil_image = Image.fromarray(image_region).convert('RGB')
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
            
            piece_type = self.classes[predicted.item()]
            confidence = probabilities[predicted.item()].item() * 100
            
            return piece_type, confidence

    def capture_clock(self, player='local'):
        """Capture the clock region for a specific player"""
        coords = self.clock_coords[player]
        screenshot = pyautogui.screenshot(region=(
            coords['top_left'][0],
            coords['top_left'][1],
            coords['bottom_right'][0] - coords['top_left'][0],
            coords['bottom_right'][1] - coords['top_left'][1]
        ))
        return np.array(screenshot)

    def check_clock_running(self, player='local'):
        """Check if a player's clock is currently running"""
        try:
            
            clock_image1 = self.capture_clock(player)
            time.sleep(0.1)  
            clock_image2 = self.capture_clock(player)
            
          
            diff = np.sum(np.abs(clock_image2 - clock_image1))
            
            
            return diff > 100 
            
        except Exception as e:
            print(f"Error checking clock: {str(e)}")
            return False

    def get_active_player(self):
        """Determine which player is currently active based on clock movement"""
        local_active = self.check_clock_running('local')
        enemy_active = self.check_clock_running('enemy')
        
        if local_active and not enemy_active:
            return 'local'
        elif enemy_active and not local_active:
            return 'enemy'
        else:
            return None 

    def detect_pieces(self, image):
        squares, positions = self.split_board_into_squares(image)
        detections = []
        
        if self.debug_mode:
            os.makedirs("debug_squares", exist_ok=True)
        
        for idx, (square, pos) in enumerate(zip(squares, positions)):
            piece_type, confidence = self.detect_piece(square)
            detections.append((*pos, piece_type, confidence))
            
            if self.debug_mode:
                square_img = Image.fromarray(square)
                square_img.save(f"debug_squares/square_{idx}_{piece_type}.png")
        
        return detections
    
    def draw_labels(self, image, detections):
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        piece_map = [['' for _ in range(8)] for _ in range(8)]
        
        for x1, y1, x2, y2, piece_type, confidence in detections:
            square_width = (x2 - x1)
            square_height = (y2 - y1)
            col = int(x1 / square_width)
            row = int(y1 / square_height)
            piece_map[row][col] = piece_type
        
        height, width = image.shape[:2]
        square_height = height // 8
        square_width = width // 8
        
        for row in range(8):
            for col in range(8):
                x1 = col * square_width
                y1 = row * square_height
                x2 = (col + 1) * square_width
                y2 = (row + 1) * square_height
                
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                
                piece = piece_map[row][col]
                if piece:
                    text_width = draw.textlength(piece, font=font)
                    text_x = x1 + (square_width - text_width) // 2
                    text_y = y1 + (square_height - 40) // 2
                    
                    text_color = "gray" if piece == "eM" else "white"
                    
                    for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
                        draw.text(
                            (text_x + offset[0], text_y + offset[1]),
                            piece,
                            fill="black",
                            font=font
                        )
                    draw.text((text_x, text_y), piece, fill=text_color, font=font)
        
        return np.array(img)
    
    def save_image(self, image):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chess_detection_{timestamp}.png"
        
        os.makedirs("detections", exist_ok=True)
        filepath = os.path.join("detections", filename)
        
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return filepath

    def generate_fen(self, detections):
        """
        Generate FEN notation from detected pieces.
        Properly handles the perspective based on which side is playing.
        """
        board = [['' for _ in range(8)] for _ in range(8)]
        
        for x1, y1, x2, y2, piece_type, _ in detections:
            square_width = (x2 - x1)
            square_height = (y2 - y1)
            col = int(x1 / square_width)
            row = int(y1 / square_height)
            board[row][col] = self.fen_mapping[piece_type]
        
        
        fen_parts = []
        for row in board:
            empty_count = 0
            row_str = ''
            
            for piece in row:
                if piece == '1':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += piece
            
            if empty_count > 0:
                row_str += str(empty_count)
            
            fen_parts.append(row_str)
        
        return '/'.join(fen_parts)

    def flip_fen(self, fen):
        """
        Flip a FEN string to represent the opposite perspective.
        This converts between black's and white's view of the board.
        """
        
        rows = fen.split('/')
        
        
        rows = rows[::-1]
        
        
        for i, row in enumerate(rows):
            
            expanded = ''
            for char in row:
                if char.isdigit():
                    expanded += '1' * int(char)
                else:
                    expanded += char
            
           
            expanded = expanded[::-1]
            
           
            compressed = ''
            count = 0
            for char in expanded:
                if char == '1':
                    count += 1
                else:
                    if count > 0:
                        compressed += str(count)
                        count = 0
                    compressed += char
            if count > 0:
                compressed += str(count)
            
            rows[i] = compressed
        
       
        return '/'.join(rows)

    def process_screen(self):
        """
        Captures the screen, detects pieces, generates labels and FEN notation
        Returns:
            saved_path (str): Path to the saved labeled image
            detections (list): List of piece detections with coordinates
            fen (str): Generated FEN notation of the board position
        """
        try:
            board = self.capture_screen()
            if board is None:
                raise ValueError("Failed to capture screen")

            detections = self.detect_pieces(board)
            if not detections:
                raise ValueError("No pieces detected on board")

            labeled_image = self.draw_labels(board, detections)
            saved_path = self.save_image(labeled_image)
            fen = self.generate_fen(detections)

            return saved_path, detections, fen

        except Exception as e:
            raise Exception(f"Error in process_screen: {str(e)}")


class ChessGUI:
    def __init__(self, detector):
        self.root = tk.Tk()
        self.root.title("Chess Position Analyzer")
        self.root.geometry("800x600")
        self.move_delay = 0.1  
        self.random_delay = 5.0  
        self.use_random_delay = False  
        
        self.detector = detector
        self.message_queue = queue.Queue()

        self.auto_scan = False
        self.auto_scan_delay = 0
        self.last_fen = None
        self.waiting_for_opponent = False
        self.after_id = None
        self.file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        self.reverse_file_map = {v: k for k, v in self.file_map.items()}
        self.blunder_chance = 3/8  
        self.miss_chance = 19/20
        
        
        self.create_gui_elements()
        
        self.stockfish = None
        try:
            stockfish_path = self.get_stockfish_path()
            if not os.path.exists(stockfish_path):
                self.log_message(f"Stockfish binary not found at: {stockfish_path}")
                self.log_message("Please download Stockfish and place it in the correct location")
            else:
                self.stockfish = Stockfish(
                    path=stockfish_path,
                    depth=16,
                    parameters={
                        "Threads": 8,
                        "Minimum Thinking Time": 0,
                        "Hash": 1024
                    }
                )
                
                try:
                    self.stockfish.get_parameters()
                    self.log_message("Stockfish initialized successfully")
                except Exception as e:
                    self.stockfish = None
                    self.log_message(f"Stockfish initialization failed: {str(e)}")
                    
        except Exception as e:
            self.stockfish = None
            self.log_message(f"Error initializing Stockfish: {str(e)}")
        
        self.side_to_move = tk.StringVar(value="white")
        
        self.setup_gui()
        self.setup_hotkey()
        self.check_message_queue()


    def count_pieces(self, fen):
        """
        Count the number of pieces in a FEN string to determine if it's an endgame
        """
        piece_part = fen.split(' ')[0]
        pieces = {
            'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
            'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0
        }
        
        for char in piece_part:
            if char in pieces:
                pieces[char] += 1
        
       
        white_material = pieces['Q'] * 9 + pieces['R'] * 5 + \
                        (pieces['B'] + pieces['N']) * 3 + pieces['P']
        black_material = pieces['q'] * 9 + pieces['r'] * 5 + \
                        (pieces['b'] + pieces['n']) * 3 + pieces['p']
        
        total_material = white_material + black_material
        total_pieces = sum(pieces.values()) - 2  
        
        return total_pieces, total_material
        

    def is_endgame(self, fen):
        """
        Determine if the position is an endgame based on:
        1. Total number of pieces (excluding kings)
        2. Total material value
        3. Presence of queens
        """
        piece_count, material_sum = self.count_pieces(fen)
        
       
        piece_part = fen.split(' ')[0]
        queens_present = 'Q' in piece_part or 'q' in piece_part
        
  
        return (piece_count <= 8 or 
                material_sum <= 20 or 
                (not queens_present and material_sum <= 14))

    def find_closest_to_zero_eval(self, moves):
        """Find the move that results in a position closest to 0 evaluation"""
        best_move = None
        closest_to_zero = float('inf')
        
        for move in moves:
            self.stockfish.make_moves_from_current_position([move])
            eval = self.stockfish.get_evaluation()
            self.stockfish.set_position([]) 
            
           
            if eval['type'] == 'mate':
                eval_value = 10000 if eval['value'] > 0 else -10000
            else:
                eval_value = eval['value']
            
            if abs(eval_value) < abs(closest_to_zero):
                closest_to_zero = eval_value
                best_move = move
                
        return best_move
        
    def check_for_material_win(self, moves):
        """Check if any move leads to a significant material advantage (>2 pawns)"""
        winning_moves = []
        
        for move in moves:
            self.stockfish.make_moves_from_current_position([move])
            eval = self.stockfish.get_evaluation()
            self.stockfish.set_position([])  
            
            
            if eval['type'] == 'cp' and eval['value'] > 200:  
                winning_moves.append((move, eval['value']))
                
        return winning_moves

    def flip_move(self, move_notation):
        """
        Flip a move's coordinates to match black's perspective.
        For example: 'e2e4' becomes 'e7e5' when viewed from black's side.
        """
        if not move_notation or len(move_notation) < 4:
            return move_notation
            
       
        source_square = move_notation[0:2]
        target_square = move_notation[2:4]
        
      
        source_file = source_square[0]
        source_rank = str(9 - int(source_square[1]))          
        
        target_file = target_square[0]
        target_rank = str(9 - int(target_square[1]))  
        
       
        flipped_move = source_file + source_rank + target_file + target_rank
        
       
        if len(move_notation) > 4:
            flipped_move += move_notation[4]
            
        return flipped_move

    def execute_move(self, move_notation):
        """Execute a chess move by controlling the mouse"""
        try:
            if not move_notation or len(move_notation) < 4:
                self.log_message("Invalid move notation")
                return False
            
            
            if self.use_random_delay:
                random_delay = random.uniform(0, self.random_delay_scale_var.get())
                self.log_message(f"Waiting {random_delay:.1f}s before moving...")
                time.sleep(random_delay)
            
            playing_as_white = self.side_to_move.get() == "white"
            source_square = move_notation[:2]
            target_square = move_notation[2:4]
            
            self.log_message(f"Moving from {source_square} to {target_square}")
            
            source_x, source_y = self.detector.get_square_center(source_square, playing_as_white)
            target_x, target_y = self.detector.get_square_center(target_square, playing_as_white)
            
           
            self.log_message(f"Screen coordinates: ({source_x}, {source_y}) to ({target_x}, {target_y})")
            
           
            pyautogui.moveTo(source_x, source_y, duration=self.move_delay)
            time.sleep(self.move_delay)
            pyautogui.mouseDown()
            time.sleep(self.move_delay)
            pyautogui.moveTo(target_x, target_y, duration=self.move_delay)
            time.sleep(self.move_delay)
            pyautogui.mouseUp()
            return True
            
        except Exception as e:
            self.log_message(f"Error executing move: {str(e)}")
            return False

    def check_stockfish_status(self):
        """Check if Stockfish is working properly"""
        if not self.stockfish:
            return False
        try:
            self.stockfish.get_parameters()
            return True
        except:
            return False
    
    def create_gui_elements(self):
        """Create basic GUI elements needed for logging"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(self.main_frame, height=15, width=60)
        self.log_text.grid(row=4, column=0, pady=5)
        
        scrollbar = ttk.Scrollbar(
            self.main_frame,
            orient=tk.VERTICAL,
            command=self.log_text.yview
        )
        scrollbar.grid(row=4, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
    
    def setup_hotkey(self):
        """Setup keyboard hotkey"""
        try:
            keyboard.on_press_key("F2", lambda _: self.message_queue.put("scan"))
            self.log_message("Hotkey (F2) registered successfully")
        except Exception as e:
            self.log_message(f"Error setting up hotkey: {str(e)}")
    
    def log_message(self, message):
        """Add a message to the log text area"""
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            print(message)
    
    def check_message_queue(self):
        """Check for messages in the queue (for hotkey handling)"""
        try:
            message = self.message_queue.get_nowait()
            if message == "scan":
                self.scan_position()
        except queue.Empty:
            pass
        self.root.after(100, self.check_message_queue)
    
    def get_stockfish_path(self):
        """Get the path to the Stockfish binary"""
        system = platform.system().lower()
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stockfish")
        
        if system == "windows":
            path = os.path.join(base_dir, "stockfish-windows-x86-64-avx2.exe")
        elif system == "linux":
            path = os.path.join(base_dir, "stockfish-ubuntu-x86-64")
        elif system == "darwin":  
            path = os.path.join(base_dir, "stockfish-macos-x86-64")
        else:
            raise Exception(f"Unsupported operating system: {system}")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def open_region_selector(self):
        BoardRegionSelector(self)
    
    def setup_gui(self):
        """Setup the complete GUI"""
        
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_var = tk.StringVar(
            value="Stockfish: " + ("Ready" if self.stockfish else "Not Available")
        )
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Button(
            status_frame,
            text="Restart Stockfish",
            command=self.restart_stockfish
        ).pack(side=tk.RIGHT)

     
        board_region_frame = ttk.Frame(self.main_frame)
        board_region_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(
            board_region_frame,
            text="Set Board Region",
            command=self.open_region_selector
        ).pack(fill=tk.X, padx=5)

       
        delay_frame = ttk.LabelFrame(self.main_frame, text="Move Speed Settings", padding="5")
        delay_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

      
        ttk.Label(delay_frame, text="Move Speed").grid(row=0, column=0, padx=5, pady=(5,0), sticky=tk.W)
        
        fixed_delay_frame = ttk.Frame(delay_frame)
        fixed_delay_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=(0,10))
        
        self.delay_var = tk.DoubleVar(value=self.move_delay)
        delay_scale = ttk.Scale(
            fixed_delay_frame,
            from_=0.1,
            to=1.0,
            variable=self.delay_var,
            orient=tk.HORIZONTAL,
            command=self.update_move_delay
        )
        delay_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0,5))
        
        self.fixed_delay_label = ttk.Label(fixed_delay_frame, text="0.3s", width=5)
        self.fixed_delay_label.grid(row=0, column=1)
        self.delay_var.trace_add(
            "write",
            lambda *args: self.fixed_delay_label.config(text=f"{self.delay_var.get():.1f}s")
        )

       
        ttk.Separator(delay_frame, orient='horizontal').grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

     
        random_delay_frame = ttk.Frame(delay_frame)
        random_delay_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=(0,5))

     
        self.random_delay_var = tk.BooleanVar(value=self.use_random_delay)
        ttk.Checkbutton(
            random_delay_frame,
            text="Use Random Delay",
            variable=self.random_delay_var,
            command=self.toggle_random_delay
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0,5))

      
        ttk.Label(random_delay_frame, text="Max Random Delay:").grid(row=1, column=0, sticky=tk.W)
        
        max_delay_control_frame = ttk.Frame(random_delay_frame)
        max_delay_control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        self.random_delay_scale_var = tk.DoubleVar(value=self.random_delay)
        random_delay_scale = ttk.Scale(
            max_delay_control_frame,
            from_=0.1,
            to=5.0,
            variable=self.random_delay_scale_var,
            orient=tk.HORIZONTAL,
            command=self.update_random_delay
        )
        random_delay_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0,5))
        
        self.random_delay_label = ttk.Label(max_delay_control_frame, text="1.0s", width=5)
        self.random_delay_label.grid(row=0, column=1)
        self.random_delay_scale_var.trace_add(
            "write",
            lambda *args: self.random_delay_label.config(text=f"{self.random_delay_scale_var.get():.1f}s")
        )
        
       
        move_frame = ttk.LabelFrame(self.main_frame, text="Side to Move", padding="5")
        move_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(
            move_frame,
            text="White",
            variable=self.side_to_move,
            value="white"
        ).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(
            move_frame,
            text="Black",
            variable=self.side_to_move,
            value="black"
        ).grid(row=0, column=1, padx=5)
        
    
        auto_scan_frame = ttk.LabelFrame(self.main_frame, text="Auto Scan", padding="5")
        auto_scan_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.auto_scan_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            auto_scan_frame,
            text="Enable Auto Scan (5s)",
            variable=self.auto_scan_var,
            command=self.toggle_auto_scan
        ).pack(side=tk.LEFT, padx=5)
        
      
        ttk.Button(
            self.main_frame,
            text="Scan Position (F2)",
            command=self.scan_position
        ).grid(row=5, column=0, pady=5)
        
       
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        results_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
     
        ttk.Label(results_frame, text="FEN:").grid(row=0, column=0, sticky=tk.W)
        self.fen_var = tk.StringVar()
        ttk.Entry(
            results_frame,
            textvariable=self.fen_var,
            width=50
        ).grid(row=0, column=1, pady=5)
        
      
        ttk.Label(results_frame, text="Best Move:").grid(row=1, column=0, sticky=tk.W)
        self.best_move_var = tk.StringVar()
        ttk.Entry(
            results_frame,
            textvariable=self.best_move_var,
            width=50
        ).grid(row=1, column=1, pady=5)
        
    
        ttk.Label(results_frame, text="Evaluation:").grid(row=2, column=0, sticky=tk.W)
        self.eval_var = tk.StringVar()
        ttk.Entry(
            results_frame,
            textvariable=self.eval_var,
            width=50
        ).grid(row=2, column=1, pady=5)


    def toggle_auto_scan(self):
        """Toggle auto-scan functionality"""
        self.auto_scan = self.auto_scan_var.get()
        if self.auto_scan:
            self.log_message("Auto-scan enabled")
           
            self.scan_and_move()
           
            self.start_auto_scan()
        else:
            self.log_message("Auto-scan disabled")
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None

    def start_auto_scan(self):
        """Start the auto-scan cycle"""
        if self.auto_scan:
           
            _, _, self.last_fen = self.detector.process_screen()
            self.log_message("Starting to scan for opponent's move")
            self.scan_for_opponent_move()

    def toggle_random_delay(self):
        """Toggle random delay functionality"""
        self.use_random_delay = self.random_delay_var.get()
        if self.use_random_delay:
          
            self.log_message(f"Random delay enabled (0.1s to {self.random_delay:.1f}s)")
        else:
            self.log_message("Random delay disabled")

    def update_random_delay(self, *args):
        """Update the random delay maximum value"""
        
        self.random_delay = self.random_delay_scale_var.get()
        if self.use_random_delay:
            self.log_message(f"Random delay updated (0.1s to {self.random_delay:.1f}s)")

    def scan_for_opponent_move(self):
        """Continuously scan for opponent's move without making moves"""
        if not self.auto_scan:
            return

        try:
           
            _, _, current_fen = self.detector.process_screen()

            if self.last_fen and current_fen != self.last_fen:
                self.log_message("Opponent move detected!")
                
                self.scan_and_move()
               
                _, _, self.last_fen = self.detector.process_screen()
                self.log_message("Resumed scanning for opponent's next move")
            
        except Exception as e:
            self.log_message(f"Auto-scan error: {str(e)}")

      
        self.after_id = self.root.after(self.auto_scan_delay, self.scan_for_opponent_move)
        
    def scan_and_move(self):
        """Scan current position and make a move, playing perfectly in endgame positions"""
        try:
            saved_path, detections, base_fen = self.detector.process_screen()
            
           
            if self.side_to_move.get() == "black":
                base_fen = self.detector.flip_fen(base_fen)
            
           
            castling_rights = self.determine_castling_rights(base_fen)
            fen = f"{base_fen} {self.side_to_move.get()[0]} {castling_rights} - 0 1"
            self.fen_var.set(fen)
            
            if not self.stockfish or not self.check_stockfish_status():
                raise Exception("Stockfish not available")

            if not self.stockfish.is_fen_valid(fen):
                raise ValueError("Invalid FEN position detected")
            
            self.stockfish.set_fen_position(fen)
            
           
            endgame = self.is_endgame(fen)
            
        
            top_moves = self.stockfish.get_top_moves(3)
            if not top_moves:
                raise ValueError("No valid moves found")
            
            if endgame:
           
                selected_move = top_moves[0]['Move']
                self.log_message("Endgame detected - playing best move")
                move_rank = "best"
            else:
              
                rand_val = random.random()
                
                if rand_val < 0.875:  
                    selected_move = top_moves[0]['Move']
                    move_rank = "best"
                elif rand_val < 0.95:  
                    if len(top_moves) > 1:
                        selected_move = top_moves[1]['Move']
                        move_rank = "second best"
                    else:
                        selected_move = top_moves[0]['Move']
                        move_rank = "best (fallback)"
                else:  
                    if len(top_moves) > 2:
                        selected_move = top_moves[2]['Move']
                        move_rank = "third best"
                    else:
                        selected_move = top_moves[-1]['Move']
                        move_rank = f"{'second' if len(top_moves) == 2 else 'best'} (fallback)"
            
          
            if selected_move:
                self.stockfish.make_moves_from_current_position([selected_move])
                evaluation = self.stockfish.get_evaluation()
                
              
                if evaluation['type'] == 'cp':
                    eval_str = f"{evaluation['value']/100:.2f}"
                else:
                    eval_str = f"Mate in {evaluation['value']}"
                self.eval_var.set(eval_str)
                
                self.best_move_var.set(selected_move)
                self.log_message(f"Selected {move_rank} move: {selected_move}")
                
                success = self.execute_move(selected_move)
                if success:
                    self.log_message("Move executed successfully")
                else:
                    self.log_message("Failed to execute move")
                    self.auto_scan = False
                    self.auto_scan_var.set(False)
            else:
                self.log_message("No valid move found")
                self.auto_scan = False
                self.auto_scan_var.set(False)
            
        except Exception as e:
            self.log_message(f"Error during scan and move: {str(e)}")
            self.auto_scan = False
            self.auto_scan_var.set(False)

    def evaluate_move_safety(self, move, current_eval):
        """
        Evaluate if a move is safe enough to be considered for a blunder.
        Returns a tuple of (is_safe, eval_diff) where:
        - is_safe: boolean indicating if the move doesn't lose too much material
        - eval_diff: evaluation difference from the best move
        """
        self.stockfish.make_moves_from_current_position([move])
        move_eval = self.stockfish.get_evaluation()
        self.stockfish.set_position([]) 
        
        
        current_cp = current_eval['value'] if current_eval['type'] == 'cp' else \
                    (10000 if current_eval['value'] > 0 else -10000)
        move_cp = move_eval['value'] if move_eval['type'] == 'cp' else \
                  (10000 if move_eval['value'] > 0 else -10000)
        
        eval_diff = current_cp - move_cp
        
       
        QUEEN_VALUE = 900  
        MAX_SAFE_LOSS = 300  
        
      
        is_safe = (
            eval_diff < QUEEN_VALUE and  
            eval_diff < MAX_SAFE_LOSS and  
            move_eval['type'] != 'mate'  
        )
        
        return is_safe, eval_diff
            
    def determine_castling_rights(self, base_fen):
        """Determine castling rights from the base FEN"""
        fen_parts = base_fen.split('/')
        castling_rights = ""
        
       
        if 'K' in fen_parts[7] and fen_parts[7].index('K') == 4:  
            if fen_parts[7].startswith('R'):  
                castling_rights += "Q"
            if fen_parts[7].endswith('R'):    
                castling_rights += "K"
        
       
        if 'k' in fen_parts[0] and fen_parts[0].index('k') == 4: 
            if fen_parts[0].startswith('r'): 
                castling_rights += "q"
            if fen_parts[0].endswith('r'):   
                castling_rights += "k"
        
        return castling_rights if castling_rights else "-"

    def update_move_delay(self, *args):
        """Update the move delay value"""
        self.move_delay = self.delay_var.get()
    
    def restart_stockfish(self):
        """Attempt to restart the Stockfish engine"""
        try:
            if self.stockfish:
                del self.stockfish
            
            stockfish_path = self.get_stockfish_path()
            if not os.path.exists(stockfish_path):
                raise FileNotFoundError(f"Stockfish binary not found at: {stockfish_path}")
                
            self.stockfish = Stockfish(
                path=stockfish_path,
                depth=20,
                parameters={
                    "Threads": 4,
                    "Minimum Thinking Time": 5,
                    "Hash": 512
                }
            )
            
            if not self.check_stockfish_status():
                raise Exception("Stockfish process failed to start")
                
            self.log_message("Stockfish engine restarted successfully")
            self.status_var.set("Stockfish: Ready")
            return True
            
        except Exception as e:
            self.log_message(f"Failed to restart Stockfish: {str(e)}")
            self.stockfish = None
            self.status_var.set("Stockfish: Not Available")
            return False
    
    def scan_position(self):
        """Scan the chess position and analyze it"""
        try:
            self.log_message("Scanning position...")
            saved_path, detections, base_fen = self.detector.process_screen()
            
           
            if self.side_to_move.get() == "black":
                base_fen = self.detector.flip_fen(base_fen)
            
           
            castling_rights = self.determine_castling_rights(base_fen)
            
        
            fen = f"{base_fen} {self.side_to_move.get()[0]} {castling_rights} - 0 1"
            self.fen_var.set(fen)
            
          
            if self.stockfish and self.check_stockfish_status():
                try:
                    if not self.stockfish.is_fen_valid(fen):
                        raise ValueError("Invalid FEN position detected")
                    
                    self.stockfish.set_fen_position(fen)
                    best_move = self.stockfish.get_best_move()
                    evaluation = self.stockfish.get_evaluation()
                    
                    self.best_move_var.set(best_move)
                    if evaluation['type'] == 'cp':
                        eval_str = f"{evaluation['value']/100:.2f}"
                    else:
                        eval_str = f"Mate in {evaluation['value']}"
                    self.eval_var.set(eval_str)
                    
                    if best_move:
                        self.log_message(f"Executing best move: {best_move}")
                        success = self.execute_move(best_move)
                        if success:
                            self.log_message("Move executed successfully")
                        else:
                            self.log_message("Failed to execute move")
                    
                except Exception as e:
                    self.log_message(f"Stockfish analysis/move execution error: {str(e)}")
                    self.best_move_var.set("Analysis failed")
                    self.eval_var.set("Error")
            else:
                self.log_message("Stockfish not available - skipping analysis and move")
                self.best_move_var.set("Stockfish not available")
                self.eval_var.set("N/A")
            
            self.log_message(f"Position scanned and saved to: {saved_path}")
            
        except Exception as e:
            self.log_message(f"Error during scanning: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    model_path = 'chess_piece_model.pth'
    detector = ChessScreenDetector(model_path)
    gui = ChessGUI(detector)
    gui.run()


if __name__ == "__main__":
    main()
