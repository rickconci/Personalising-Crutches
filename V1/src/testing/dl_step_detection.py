import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments

class StepPatchDataset(Dataset):
    """
    Creates a dataset of windowed signal patches for step detection.
    Each patch is labeled as 1 if it contains a ground truth step, and 0 otherwise.
    """
    def __init__(self, signals, times, gt_steps,
                 win_sec=2.0, stride_sec=1.0, fs=100.0):
        self.X, self.y = [], []
        w, s = int(win_sec*fs), int(stride_sec*fs)
        for start in range(0, len(signals)-w, s):
            end   = start + w
            patch = signals[start:end, :]
            # Label is true if any ground truth step falls within the window time range
            label = ((gt_steps >= times[start]) & (gt_steps < times[end])).any()
            self.X.append(patch)
            self.y.append(float(label))
        self.X = np.stack(self.X).astype('float32')
        self.y = np.array(self.y).astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # TimesFM expects (batch, sequence_length, num_features)
        return self.X[idx], self.y[idx]


class TimeSFMStepDetector:
    """
    A class to handle fine-tuning and prediction using a Time Series Foundation Model (TimesFM)
    for step detection.
    """
    def __init__(self, model_name="google/timesfm-2.0-500m-pytorch"):
        self.model_name = model_name

    def train(self, multichannel_signal, times, gt_steps, fs,
              output_dir="finetuned_step_fm",
              win_sec=2.0, stride_sec=1.0,
              epochs=10, lr=2e-4, batch_size=32,
              freeze_layers=0):
        """
        Fine-tunes the TimesFM model for step detection classification.
        """
        train_ds = StepPatchDataset(multichannel_signal,
                                    times,
                                    gt_steps,
                                    win_sec=win_sec,
                                    stride_sec=stride_sec,
                                    fs=fs)
        
        num_input_features = multichannel_signal.shape[1]
        context_length = int(win_sec * fs)

        cfg = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=1,
            problem_type="single_label_classification",
            num_input_features=num_input_features,
            context_length=context_length
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            config=cfg, 
            ignore_mismatched_sizes=True
        )

        if freeze_layers > 0:
            # Freeze the first N encoder layers of the model
            if hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
                for i, layer in enumerate(model.model.encoder.layers):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"Froze first {freeze_layers} encoder layers.")
            else:
                print("Warning: Model structure not as expected for freezing layers. Could not freeze layers.")

        args = TrainingArguments(
            output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            save_strategy="epoch",
            logging_steps=10,
            fp16=True if torch.cuda.is_available() else False,
        )
        
        trainer = Trainer(model, args, train_dataset=train_ds)
        trainer.train()
        trainer.save_model(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}")

    def predict(self, multichannel_signal, processed_signal_for_refinement, times, fs,
                ckpt="finetuned_step_fm",
                win_sec=2.0, stride_sec=0.25, threshold=0.9):
        """
        Predicts step times using a fine-tuned TimesFM model.
        """
        w = int(win_sec * fs)
        s = int(stride_sec * fs)
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
            if torch.cuda.is_available():
                model.to('cuda')
        except Exception as e:
            print(f"Error loading model from {ckpt}: {e}")
            return np.array([])

        sig = multichannel_signal
        preds = []
        
        with torch.no_grad():
            for start in range(0, len(sig) - w, s):
                segment = torch.tensor(sig[start:start+w, :]).unsqueeze(0).float()
                if torch.cuda.is_available():
                    segment = segment.to('cuda')
                    
                p = torch.sigmoid(model(segment).logits)[0, 0]
                if p > threshold:
                    # Refine step location by finding the peak in the original processed signal
                    local_idx = np.argmax(processed_signal_for_refinement[start:start+w])
                    step_time = times[start + local_idx]
                    preds.append(step_time)

        if not preds:
            return np.array([])
            
        # Post-process predictions to remove duplicates and enforce a minimum interval
        preds = np.sort(np.unique(preds))
        unique_preds = []
        if len(preds) > 0:
            last_pred = -np.inf
            min_interval = 0.2 # seconds
            for p in preds:
                if p - last_pred > min_interval:
                    unique_preds.append(p)
                    last_pred = p
        return np.array(unique_preds) 