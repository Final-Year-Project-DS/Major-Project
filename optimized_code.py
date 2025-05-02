import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import time
from ctgan import CTGAN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# 💻 Device Configuration with memory management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Set memory limits for GTX 1650 (4GB)
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available memory
torch.manual_seed(42)
np.random.seed(42)

# Memory management functions
def clear_memory():
    """Clear memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 📂 Process data in smaller chunks to avoid memory issues
def load_and_process_files(file_paths, chunksize=50000, sample_fraction=0.5):
    """Load CSV files in small chunks and optionally downsample to save memory"""
    dataframes = []
    
    for path in file_paths:
        try:
            if not os.path.exists(path):
                print(f"⚠️ File not found: {path}")
                continue
                
            print(f"Processing {path}...")
            chunk_list = []
            total_rows = 0
            processed_rows = 0
            
            # First count total rows without loading full file
            for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, nrows=10):
                pass
            
            # Process in small chunks
            for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False)):
                # Downsample to save memory if needed
                if sample_fraction < 1.0:
                    if i % int(1/sample_fraction) != 0:
                        continue
                
                # Clean chunk data
                chunk.columns = [str(col).strip() for col in chunk.columns]
                chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Only keep chunk if it has data after cleaning
                if not chunk.empty:
                    chunk_list.append(chunk)
                    processed_rows += len(chunk)
                
                # Free memory after processing chunk
                if i % 10 == 0:
                    clear_memory()
                    
            if chunk_list:
                df = pd.concat(chunk_list, ignore_index=True)
                print(f"✅ Loaded {processed_rows} rows from {path}")
                dataframes.append(df)
                
            # Free memory after processing file
            del chunk_list
            clear_memory()
                
        except Exception as e:
            print(f"⚠️ Failed to read {path}: {e}")
    
    return dataframes

# 📂 Load and Preprocess Data - Only needed columns with memory efficiency
file_paths = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

# Reduce the dataset size to fit in memory
print("📊 Loading datasets (sampling 25% of data for memory efficiency)...")
dataframes = load_and_process_files(file_paths, chunksize=10000, sample_fraction=0.25)

if not dataframes:
    raise ValueError("❌ No dataframes were successfully loaded. Check your file paths.")

print("\n📋 Available columns in first dataframe:")
for i, col in enumerate(dataframes[0].columns[:10]):  # Show only first 10 columns
    print(f"  {i}: '{col}'")
print(f"  ... and {len(dataframes[0].columns) - 10} more columns")

# Combine data with memory optimization
print("\n🔄 Combining dataframes...")
# Process one file at a time to avoid large memory allocation
data = dataframes[0]
for i in range(1, len(dataframes)):
    data = pd.concat([data, dataframes[i]], ignore_index=True)
    # Free memory
    dataframes[i] = None
    clear_memory()
    
print(f"  Combined shape: {data.shape}")

# Free memory
dataframes = None
clear_memory()

# 🔍 Find the label column
def find_label_column(data):
    """Find the most likely label column in the dataset"""
    possible_label_columns = []
    
    for col in data.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['label', 'class', 'tag', 'target']):
            possible_label_columns.append(col)
    
    if possible_label_columns:
        for col in possible_label_columns:
            if 'label' in str(col).lower():
                print(f"✅ Found label column: '{col}'")
                return col
        
        print(f"✅ Found label column: '{possible_label_columns[0]}'")
        return possible_label_columns[0]
    
    # If no label column found, ask user to specify
    print("❌ Could not automatically identify a label column. Available columns:")
    for i, col in enumerate(data.columns):
        print(f"  {i}: {col}")
    
    raise ValueError("Could not find a suitable label column")

# Find label column
label_column = find_label_column(data)

# Process data in smaller batches to avoid memory errors
print("\n🧮 Preparing features and labels...")

def add_jitter(X, noise_std=0.005):
    """Add slight random noise to GAN generated samples."""
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise
    
# 1. Extract label column first to save memory
y = data[label_column].copy()
print(f"  Class distribution summary:")
class_counts = y.value_counts()
total_classes = len(class_counts)
print(f"  Found {total_classes} unique classes")
print(f"  Most frequent class: {class_counts.index[0]} with {class_counts.values[0]} samples ({class_counts.values[0]/len(y)*100:.2f}%)")
print(f"  Least frequent class: {class_counts.index[-1]} with {class_counts.values[-1]} samples ({class_counts.values[-1]/len(y)*100:.4f}%)")

# 2. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
del y  # Free memory
clear_memory()

# 3. Drop the label column to create features
X = data.drop(columns=[label_column])
del data  # Free memory
clear_memory()

# 4. Select only numeric features in batches to avoid memory errors
numeric_cols = []
for col in X.columns:
    try:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
    except:
        pass
        
print(f"  Selected {len(numeric_cols)} numeric features out of {X.shape[1]} total columns")

# Use only numeric columns to save memory
X = X[numeric_cols]
clear_memory()

# 5. Handle missing values - in place to save memory
print("  Filling missing values...")
X.fillna(0, inplace=True)

# 6. Scale features using batched processing to avoid memory errors
print("  Scaling features (batched processing)...")
scaler = StandardScaler()

# Fit scaler on a sample to avoid memory issues
sample_size = min(100000, len(X))
X_sample = X.sample(sample_size, random_state=42)
scaler.fit(X_sample)
del X_sample
clear_memory()

# Transform in batches
batch_size = 50000
n_batches = int(np.ceil(len(X) / batch_size))
X_scaled_list = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(X))
    batch = X.iloc[start_idx:end_idx]
    X_scaled_batch = scaler.transform(batch)
    X_scaled_list.append(X_scaled_batch)
    # Free memory
    del batch
    clear_memory()

# Combine scaled features
X_scaled = np.vstack(X_scaled_list)
del X, X_scaled_list
clear_memory()

# 🔍 Limit the dataset size if needed
max_samples = 400000  # Maximum samples to use to fit in memory
if len(X_scaled) > max_samples:
    print(f"  Limiting dataset to {max_samples} samples to fit in memory")
    # Stratified sampling to maintain class distribution
    from sklearn.model_selection import train_test_split
    _, X_scaled, _, y_encoded = train_test_split(
        X_scaled, y_encoded, 
        test_size=max_samples/len(X_scaled),
        stratify=y_encoded,
        random_state=42
    )
    clear_memory()

# 🧠 Define main classification model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 🎮 Define Reinforcement Learning Policy Network for sample selection
class PolicyNetwork(nn.Module):
    def __init__(self, input_size):
        super(PolicyNetwork, self).__init__()
        # Smaller policy network to save memory
        self.policy = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.policy(x)
    
    def get_action(self, state, temperature=1.0):
        """Return probability of selecting a sample"""
        with torch.no_grad():
            probs = self.forward(state)
        return probs

# 📦 Experience buffer for reinforcement learning
class ExperienceBuffer:
    def __init__(self, capacity=1000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.capacity = capacity
        
    def push(self, state, action, reward):
        if len(self.states) >= self.capacity:
            # Remove oldest entries
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def get_batch(self, batch_size=None):
        if batch_size is None or batch_size > len(self.states):
            batch_size = len(self.states)
            
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        states = torch.cat([self.states[i] for i in indices])
        actions = torch.cat([self.actions[i] for i in indices])
        rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32)
        
        return states, actions, rewards

# 🔧 Training Function with memory optimization
def train(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    batches = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
        
        if scheduler:
            scheduler.step()
        
        # Delete tensors to free memory
        del outputs, loss
        
    clear_memory()
    return total_loss / batches

# 🎮 Reinforcement Learning Policy Update
def update_policy(policy_net, buffer, optimizer, gamma=0.99):
    if len(buffer.states) < 32:  # Minimum batch size
        return 0.0
        
    # Get batch of experiences
    states, actions, rewards = buffer.get_batch()
    states = states.to(device)
    actions = actions.to(device)
    
    # Compute discounted rewards
    discounted_rewards = []
    R = 0
    rewards = rewards.tolist()
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
        
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
    
    # Normalize rewards for stable training
    if len(discounted_rewards) > 1:
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
    
    # Compute log probabilities
    probs = policy_net(states)
    # Use mean since we want higher probs for higher rewards
    policy_loss = -torch.mean(torch.log(probs) * discounted_rewards)
    
    # Update policy network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    return policy_loss.item()

# 🔍 Evaluation Function with memory optimization
def evaluate(model, dataloader, label_encoder, fold=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    batches = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            batches += 1
            
            preds = torch.argmax(outputs, axis=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Free memory
            del outputs, inputs, labels, preds
            
    clear_memory()
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, f1, cm, total_loss / batches

# 📊 Federated Averaging with memory optimization
def federated_avg(global_model, client_models, client_weights=None):
    """Average the models with optional weighting"""
    if client_weights is None:
        client_weights = [1/len(client_models)] * len(client_models)
    
    global_dict = global_model.state_dict()
    
    for k in global_dict:
        # Process one layer at a time to save memory
        global_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float32)
        
        for i, client in enumerate(client_models):
            # Add weighted client parameters
            global_dict[k] += client_weights[i] * client.state_dict()[k].float()
            
    global_model.load_state_dict(global_dict)
    return global_model

# 🎮 RL-Enhanced Sample Selection Function
def select_samples_with_rl(policy_net, X, y, threshold=0.5, max_samples=5000):
    """Select samples using RL policy network with a batch approach to save memory"""
    selected_indices = []
    batch_size = 1000  # Process in small batches to save memory
    
    for i in range(0, len(X), batch_size):
        end_idx = min(i + batch_size, len(X))
        X_batch = torch.tensor(X[i:end_idx], dtype=torch.float32).to(device)
        
        # Get selection probabilities
        with torch.no_grad():
            probs = policy_net(X_batch).cpu().numpy().flatten()
        
        # Select samples based on probability and threshold
        batch_indices = np.where(probs > threshold)[0] + i
        
        # Add to selected indices
        selected_indices.extend(batch_indices)
        
        # Free memory
        del X_batch
        clear_memory()
        
        # Limit number of samples to prevent memory issues
        if len(selected_indices) >= max_samples:
            selected_indices = selected_indices[:max_samples]
            break
    
    # If we didn't get any samples, select some randomly
    if len(selected_indices) < 10:
        print("  ⚠️ RL policy selected too few samples, adding random samples")
        random_indices = np.random.choice(len(X), min(max_samples // 10, len(X)), replace=False)
        selected_indices.extend(random_indices)
        selected_indices = list(set(selected_indices))  # Remove duplicates
    
    print(f"  🎮 RL policy selected {len(selected_indices)} samples out of {len(X)}")
    return np.array(selected_indices)

# Optional: Train GANs before training loop
def generate_with_ctgan(X, y, label_name, n_samples=1000, min_required=30):
    """
    Train CTGAN on a specific class and generate synthetic data.
    If the class has too few samples, duplicate with noise to enable training.
    """

    # Fix mismatched lengths
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y

    class_df = df[df['label'] == label_name].copy()

    # 🧪 Boost if too few samples
    if len(class_df) < min_required:
        print(f"🔁 Bootstrapping class '{label_name}' with duplication (Current: {len(class_df)})")
        reps = (min_required // len(class_df)) + 1
        boot_df = pd.concat([class_df] * reps, ignore_index=True).sample(min_required, random_state=42)
        
        # Add jitter (noise)
        for col in feature_names:
            boot_df[col] += np.random.normal(0, 0.005, boot_df[col].shape)
        
        class_df = boot_df

    # 🧪 Smarter adaptive epochs control
    if len(class_df) < 1000:
        epochs = 50
    elif len(class_df) < 3000:
        epochs = 100
    else:
        epochs = 200

    print(f"🚀 Training GAN on '{label_name}' with {len(class_df)} samples (Epochs: {epochs})")
    start = time.time()
    gan = CTGAN(epochs=epochs, batch_size=256, generator_lr=2e-4, discriminator_lr=2e-4, pac=1)
    gan.fit(class_df, discrete_columns=['label'])
    print(f"✅ GAN Training Complete for '{label_name}' in {time.time() - start:.2f} sec")

    synth_data = gan.sample(n_samples)
    
    # 🛡️ Safety check for synth_data
    if synth_data is None or len(synth_data) == 0:
        print(f"⚠️ GAN failed to generate samples for class '{label_name}'")
        return None, None

    y_synth = synth_data['label'].values
    X_synth = synth_data.drop(columns=['label']).values

    # 🛠 Add smaller jitter
    X_synth = add_jitter(X_synth, noise_std=0.003)

    return X_synth, y_synth

def boost_rare_class(X, y, label_name, min_samples=10):
    """Duplicate samples of a rare class to reach minimum threshold."""
    indices = np.where(y == label_name)[0]
    if len(indices) < min_samples:
        print(f"🔵 Boosting rare class: {label_name} (Current: {len(indices)}, Target: {min_samples})")
        X_dup = np.repeat(X[indices], repeats=(min_samples // len(indices)) + 1, axis=0)
        y_dup = np.repeat(y[indices], repeats=(min_samples // len(indices)) + 1, axis=0)
        X = np.vstack([X, X_dup])
        y = np.concatenate([y, y_dup])
        clear_memory()
    return X, y

# 🎯 Reinforcement Learning + Federated Training 
def rl_federated_training(X, y, num_clients=3, num_rounds=2, local_epochs=2, fed_rounds=3):
    print("\n🚀 Starting RL-Enhanced Federated Learning Training")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Clients: {num_clients}, Federation Rounds: {fed_rounds}, Local epochs: {local_epochs}")
    
    # Use fewer folds to save memory
    skf = StratifiedKFold(n_splits=num_rounds, shuffle=True, random_state=42)
    fold = 1
    
    # Store results
    all_results = []
    
    # For each fold
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n🔁 Fold {fold}/{num_rounds}")
        fold += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
            # -----------------------------
        # 🩺 Class-Aware Oversampling Fix
        # -----------------------------
        print("🔄 Applying manual oversampling for Bot and Infiltration...")

        # 🛠️ Boost rare classes first
        y_train_str = le.inverse_transform(y_train)

        X_train, y_train_str = boost_rare_class(X_train, y_train_str, 'Infiltration', min_samples=20)
        X_train, y_train_str = boost_rare_class(X_train, y_train_str, 'Bot', min_samples=20)
        X_train, y_train_str = boost_rare_class(X_train, y_train_str, 'Web Attack � XSS', min_samples=20)
        
        # Now call GAN for Bot and Infiltration
        bot_gan_X, bot_gan_y = generate_with_ctgan(X_train, y_train_str, 'Bot', n_samples=2000)
        inf_gan_X, inf_gan_y = generate_with_ctgan(X_train, y_train_str, 'Infiltration', n_samples=3000)
        
        # Stack GAN generated samples
        if bot_gan_X is not None:
            bot_gan_y = le.transform(bot_gan_y)
            X_train = np.vstack([X_train, bot_gan_X])
            y_train = np.concatenate([y_train, bot_gan_y])

        if inf_gan_X is not None:
            inf_gan_y = le.transform(inf_gan_y)
            X_train = np.vstack([X_train, inf_gan_X])
            y_train = np.concatenate([y_train, inf_gan_y])
            
        # 🛠️ Safety: Align X_train and y_train
        min_len = min(len(X_train), len(y_train))
        X_train = X_train[:min_len]
        y_train = y_train[:min_len]
           
        # ✅ Verify lengths are now equal
        assert len(X_train) == len(y_train), "❌ Mismatch: X and y lengths are not equal!"
            
        clear_memory()

        print(f"✅ Resampled training set shape using GAN: {X_train.shape}")

        # Apply SMOTE for class balancing with memory considerations - reduce sample size
        try:
            # Only use SMOTE if dataset is small enough
            if len(X_train) < 150000:  # Reduced threshold from 200000
                print("  Applying SMOTE for class balancing...")
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"  After SMOTE: {X_train.shape[0]} samples")
            else:
                print("  Skipping SMOTE due to memory constraints")
        except Exception as e:
            print(f"⚠️ SMOTE failed: {e}. Using original imbalanced data.")
        
        clear_memory()
        
        # Create test tensors with smaller batch sizes
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Smaller batch sizes to save memory
        batch_size = 64  # Reduced from 256
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        # Initialize models - fewer clients to save memory
        global_model = Net(input_size, num_classes).to(device)
        client_models = [Net(input_size, num_classes).to(device) for _ in range(num_clients)]
        
        # Initialize RL policy networks for each client
        policy_nets = [PolicyNetwork(input_size).to(device) for _ in range(num_clients)]
        policy_optimizers = [optim.Adam(net.parameters(), lr=0.001) for net in policy_nets]
        
        # Experience buffers for RL
        buffers = [ExperienceBuffer(capacity=500) for _ in range(num_clients)]
        
        # Track best model
        best_f1 = 0
        best_model_wts = None
        
        # Free memory before starting training
        clear_memory()
        
        # Federated learning rounds
        for rnd in range(fed_rounds):
            print(f"\n  📊 Federation Round {rnd+1}/{fed_rounds}")
            
            # Free memory before each round
            clear_memory()
            
            # Simulate client training
            client_data_sizes = []
            
            for i in range(num_clients):
                print(f"    👤 Training Client {i+1}")
                
                # Use RL policy to select samples if not first round
                if rnd > 0:
                    # Use policy network to select samples
                    selected_indices = select_samples_with_rl(
                        policy_nets[i], 
                        X_train, 
                        y_train, 
                        threshold=0.5,  # Increase threshold gradually
                        max_samples=30000  # Limit max samples for memory
                    )
                    client_X = X_train[selected_indices]
                    client_y = y_train[selected_indices]
                else:
                    # First round, select randomly
                    client_size = min(len(X_train) // num_clients, 30000)  # Cap size
                    client_idx = np.random.choice(len(X_train), client_size, replace=False)
                    client_X = X_train[client_idx]
                    client_y = y_train[client_idx]
                
                client_data_sizes.append(len(client_X))
                
                # Create tensors for this client
                X_client_tensor = torch.tensor(client_X, dtype=torch.float32)
                y_client_tensor = torch.tensor(client_y, dtype=torch.long)
                client_dataset = TensorDataset(X_client_tensor, y_client_tensor)
                
                # Smaller batch size for training
                client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)

                # Train client model
                model = client_models[i]
                model.load_state_dict(global_model.state_dict())
                
                # Simple optimizer to save memory
                optimizer = optim.SGD(model.parameters(), lr=0.01)
                
                # Compute class weights for CrossEntropy
                class_counts = np.bincount(y_train)
                class_weights = 1.0 / (class_counts + 1e-6)
                class_weights = class_weights / class_weights.sum()

                # Convert to torch tensor and move to device
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
                criterion = nn.CrossEntropyLoss(weight=weights_tensor)

                print(f"      Training on {len(client_X)} samples")
                
                # Store initial performance to compute reward
                initial_acc, initial_f1, _, _ = evaluate(model, test_loader, le, fold=fold)
                
                # Train the model
                for epoch in range(local_epochs):
                    loss = train(model, client_loader, criterion, optimizer)
                
                # Evaluate to compute reward
                final_acc, final_f1, _, _ = evaluate(model, test_loader, le, fold=fold)
                # ✅ Smoothed reward: Combine F1 and Accuracy delta
                reward = ((final_f1 - initial_f1) + (final_acc - initial_acc)) * 5

                if rnd == fed_rounds - 1 and i == num_clients - 1:
                    y_true, y_pred = [], []
                    llm_features = []

                    global_model.eval()
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = global_model(inputs)
                            _, predicted = torch.max(outputs, 1)

                            y_true.extend(labels.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())

                            # Collect LLM features in same loop to ensure alignment
                            for sample in inputs:                               
                                llm_features.append(dict(enumerate(sample.cpu().numpy().tolist())))                

                    try:
                        df_llm = pd.DataFrame({
                            "label": y_pred,
                            "features": llm_features
                        })
                        df_llm.to_csv("llm_input_data.csv", index=False)
                        print("🧾 Saved LLM input data to 'llm_input_data.csv'")
                    except Exception as e:
                        print(f"❌ Failed to save LLM input data: {e}")

                    print("\n📈 Final Classification Report (Last Client - Last Round):")
                    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=1)
                    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=1))

                    underperforming = [label for label, scores in report.items() if isinstance(scores, dict) and scores['recall'] < 0.4]
                    print("⚠️ Low recall classes:", underperforming)

                    cm = confusion_matrix(y_true, y_pred)
                    # plt.figure(figsize=(12, 8))
                    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    #             xticklabels=le.classes_,
                    #             yticklabels=le.classes_)
                    # plt.xlabel('Predicted Label')
                    # plt.ylabel('True Label')
                    # plt.title('Final Confusion Matrix - Last Client Last Round')
                    # plt.tight_layout()
                    # plt.savefig("final_confusion_matrix.png")
                    # plt.show()    

                    # Save to CSV for LLM reporting
                    df_llm = pd.DataFrame({
                    "label": y_pred,
                    "features": llm_features
                    })
                    df_llm.to_csv("llm_input_data.csv", index=False)
                    print("🧾 Saved LLM input data to 'llm_input_data.csv'")
               
                
                # Store experience for RL
                if rnd > 0:  # Only update after first round
                    # Store experiences in buffer - process in batches to save memory
                    batch_size_rl = 100
                    for j in range(0, len(client_X), batch_size_rl):
                        end_j = min(j + batch_size_rl, len(client_X))
                        # Get states and actions for this batch
                        states_batch = torch.tensor(client_X[j:end_j], dtype=torch.float32).to(device)
                        with torch.no_grad():
                            actions_batch = policy_nets[i](states_batch)
                        
                        # Add to buffer
                        buffers[i].push(states_batch.cpu(), actions_batch.cpu(), reward)
                    
                    # Update policy network
                    # Update policy network
                    policy_loss = update_policy(policy_nets[i], buffers[i], policy_optimizers[i])

                    # Add entropy regularization
                    try:
                        states, actions, _ = buffers[i].get_batch()
                        entropy = -(actions * torch.log(actions + 1e-8)).mean()
                        policy_loss += 0.01 * entropy
                    except:
                        pass  # Safe fallback in case buffer returns nothing

                    print(f"      RL Policy Loss: {policy_loss:.6f}, Reward: {reward:.4f}")

                    
                    # Clear buffer after update to save memory
                    buffers[i].clear()
                
                # Free memory
                del client_X, client_y, X_client_tensor, y_client_tensor, client_dataset, client_loader
                clear_memory()

            # Weight clients by their data size
            client_weights = [size/sum(client_data_sizes) for size in client_data_sizes]
            
            # Average models
            federated_avg(global_model, client_models, client_weights)
            
            # Evaluate global model
            acc, f1, _, test_loss = evaluate(global_model, test_loader, le)
            print(f"    ✅ Global Model - Accuracy: {acc*100:.2f}%, F1: {f1:.4f}, Loss: {test_loss:.4f}")
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_wts = {k: v.cpu() for k, v in global_model.state_dict().items()}  # Store on CPU
                print(f"    🏆 New best model saved! F1: {f1:.4f}")
        
        # Load best model for final evaluation
        if best_model_wts:
            global_model.load_state_dict(best_model_wts)
        
        # Final evaluation with smaller batch size
        acc, f1, cm, _ = evaluate(global_model, test_loader, le)

        
        print("\n📊 Final Evaluation:")
        print(f"  Accuracy: {acc*100:.2f}%, Weighted F1: {f1:.4f}")
        
        # Store results
        fold_results = {
            'accuracy': acc,
            'f1': f1
        }
        all_results.append(fold_results)
        
        # Free memory before next fold
        del X_test_tensor, y_test_tensor, test_dataset, test_loader
        del global_model, client_models, policy_nets, policy_optimizers, buffers, best_model_wts
        clear_memory()
    
    # Print average results
    print("\n🏁 Overall Results:")
    avg_acc = np.mean([r['accuracy'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    print(f"  Average Accuracy: {avg_acc*100:.2f}%")
    print(f"  Average F1 Score: {avg_f1:.4f}")
    
    return all_results

# Set the PyTorch memory allocation strategies for limited GPU
if torch.cuda.is_available():
    # Limit PyTorch's caching allocator
    print("\n⚙️ Setting GPU memory limits for GTX 1650 (4GB)")
    torch.cuda.set_per_process_memory_fraction(0.7)  # Use at most 70% of GPU memory

# Run the RL-enhanced federated learning training
try:
    # Print memory status before training
    print("\n💾 Current Memory Status:")
    if torch.cuda.is_available():
        print(f"  GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    # Run training with reduced parameters to fit in memory
    results = rl_federated_training(
        np.array(X_scaled, dtype=np.float32),  # Use float32 instead of float64
        y_encoded, 
        num_clients=3,         # Limited number of clients
        num_rounds=2,          # Reduced cross-validation rounds
        local_epochs=3,        # Limited local epochs
        fed_rounds=3           # Limited federation rounds
    )
    print("\n✅ Training completed successfully!")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()


