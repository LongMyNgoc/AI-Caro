import numpy as np
import cv2
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.losses import MeanSquaredError
import os

# Kích thước bàn cờ
Ox, Oy = 15, 15

# Lớp DQLAgent để xử lý AI
class DQLAgent:
    def __init__(self, state_size, action_size, player_id):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id  # Xác định AI là người chơi X hoặc O
        self.memory = []
        self.memory_limit = 10000000
        self.gamma = 0.95  # Tỷ lệ giảm dần giá trị
        self.epsilon = 1.0  # Xác suất khám phá (exploration)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Tỷ lệ giảm epsilon
        self.model = self._build_model()
        self.target_model = self._build_model()  # Mô hình target
        self.target_model.set_weights(self.model.get_weights())  # Khởi tạo trọng số target model

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def act(self, state):  
        # Find all valid actions (positions that are empty)  
        valid_actions = [i for i, val in enumerate(state.flatten()) if val == 0]  
        
        if not valid_actions:  
            # No valid actions left, it's a draw  
            return None  
        
        if np.random.rand() <= self.epsilon:  
            # Exploration: randomly choose a valid action  
            return random.choice(valid_actions)  
        else:  
            # Exploitation: choose the best valid action based on predicted Q-values  
            q_values = self.model.predict(state.reshape(1, -1))[0]  
            
            # Mask invalid actions by setting their Q-values to negative infinity  
            masked_q_values = np.full_like(q_values, -np.inf)  
            masked_q_values[valid_actions] = q_values[valid_actions]  
            
            return np.argmax(masked_q_values)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1))[0])  # Double DQN
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    def save(self, filename):
        filename = filename.replace('.h5', '.keras')
        self.model.save(filename)

    def load(self, filename):
        if os.path.exists(filename):
            self.model = load_model(filename, custom_objects={'mean_squared_error': MeanSquaredError()})
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            self.target_model.set_weights(self.model.get_weights())  # Cập nhật target model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # Cập nhật mô hình target sau mỗi episode

# Hàm reset bàn cờ
def reset_board():
    return np.zeros((Ox, Oy))

# Hàm thực hiện hành động của người chơi
def take_action(state, action, player_id):  
    next_state = np.copy(state)  
    
    if action is None:  
        # No valid actions left, it's a draw  
        reward = 0  
        done = True  
        return next_state, reward, done  

    x, y = divmod(action, Ox)  
    
    if next_state[x, y] == 0:  
        next_state[x, y] = player_id  
        won = check_win(next_state, x, y)  
        if won:  
            reward = 1  
            done = True  
        elif np.all(next_state != 0):  
            # Board is full and no winner, it's a draw  
            reward = 0  
            done = True  
        else:  
            reward = 0  
            done = False  
    else:  
        # Should not reach here as agents avoid invalid moves  
        reward = -1  
        done = True  # End game on invalid move for safety  

    return next_state, reward, done

# Hàm kiểm tra người thắng
def check_win(state, x, y):  
    player = state[x, y]  

    def count_in_direction(dx, dy):  
        count = 0  
        nx, ny = x + dx, y + dy  # Start from the next position  
        while 0 <= nx < Ox and 0 <= ny < Oy and state[nx, ny] == player:  
            count += 1  
            nx += dx  
            ny += dy  
        return count  

    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:  
        count = 1  # Including the current piece  
        count += count_in_direction(dx, dy)  
        count += count_in_direction(-dx, -dy)  

        if count == 5:  
            return True  

    return False

# Hàm render bàn cờ
def render_board(state, winner_id=None):  
    board_img = np.zeros((Ox * 40, Oy * 40, 3), dtype=np.uint8)  
    for i in range(Ox):  
        for j in range(Oy):  
            if state[i, j] == 1:  
                color = (0, 0, 255)  # Red for X  
            elif state[i, j] == 2:  
                color = (255, 0, 0)  # Blue for O  
            else:  
                color = (255, 255, 255)  # White for empty  
            cv2.rectangle(board_img, (j * 40, i * 40), ((j + 1) * 40, (i + 1) * 40), color, -1)  
    if winner_id is not None:  
        text = f"Player {winner_id} Wins!"  
    elif np.all(state != 0):  
        text = "It's a Draw!"  
    else:  
        text = ""  
    if text:  
        cv2.putText(board_img, text, (10, Ox * 40 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
    return board_img

# Hàm chơi game
def play_game(agent_X, agent_O, num_episodes):
    for e in range(num_episodes):
        state = reset_board()
        done = False
        while not done:
            for agent in [agent_X, agent_O]:  # Lần lượt để mỗi Agent thực hiện nước đi
                action = agent.act(state)
                next_state, reward, done = take_action(state, action, agent.player_id)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    agent.replay(min(64, len(agent.memory)))
                    agent.update_target_model()  # Cập nhật target model sau mỗi episode
                    break

# Hàm ghi lại video của trận đấu
def record_game(agent_X, agent_O):  
    state = reset_board()  
    done = False  
    frames = []  
    winner_id = None  # Track the winner  
    
    while not done:  
        for agent in [agent_X, agent_O]:  
            action = agent.act(state)  
            next_state, _, done = take_action(state, action, agent.player_id)  
            state = next_state  
            
            if done:  
                # Determine the winner or if it's a draw  
                if action is not None and check_win(state, *divmod(action, Ox)):  
                    winner_id = agent.player_id  
                else:  
                    winner_id = None  # It's a draw  
                frame = render_board(state, winner_id)  
                frames.append(frame)  
                break  
            else:  
                frame = render_board(state)  
                frames.append(frame)  
    
    if frames:  
        height, width, layers = frames[0].shape  
    else:  
        print("Error: No frames to record")  
        return  
    
    try:  
        video = cv2.VideoWriter('game_record.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (width, height))  
    except Exception as e:  
        print(f"Error initializing video writer: {e}")  
        return  
    
    for frame in frames:  
        video.write(frame)  
    
    # Hold the final frame for a few seconds  
    for _ in range(5 * 1):  
        video.write(frames[-1])  
    
    video.release()  
    
    print("Game recorded successfully!")

# Khởi tạo AI cho người chơi X và O
agent_X = DQLAgent(state_size=Ox * Oy, action_size=Ox * Oy, player_id=1)
agent_O = DQLAgent(state_size=Ox * Oy, action_size=Ox * Oy, player_id=2)

model_filename_X = 'dql_model_X.keras'
model_filename_O = 'dql_model_O.keras'

# Tải mô hình đã được huấn luyện trước
agent_X.load(model_filename_X)
agent_O.load(model_filename_O)

# Chơi game với số lượng episode xác định
play_game(agent_X, agent_O, num_episodes=10)

# Lưu lại mô hình sau khi huấn luyện
agent_X.save(model_filename_X)
agent_O.save(model_filename_O)

# Ghi lại video của trận đấu
record_game(agent_X, agent_O) 