# کد حرفه‌ای و یکپارچه برای آموزش مدل یادگیری ماشین و تجزیه و تحلیل داده‌های اندازه‌گیری بدن - نسخه پیشرفته
# بر پایه مفهوم بسته Dart body_analyzer، اما در Python با استفاده از numpy, pandas, torch و matplotlib
# بهبودها: 
# - حذف وابستگی به sklearn (پیاده‌سازی دستی StandardScaler و LabelEncoder و train_test_split)
# - افزودن کراس-والیدیشن، ماتریس درهم‌ریختگی، و معیارهای بیشتر (Precision, Recall, F1)
# - نمودارهای اضافی: همبستگی heatmap، باکس‌پلات برای توزیع‌ها، و منحنی یادگیری
# - پیش‌بینی تعاملی برای ورودی کاربر
# - ذخیره مدل و داده‌ها با نسخه‌بندی
# - لاگینگ پیشرفته و مدیریت خطا
# فرض: محیط Python 3.12 با numpy, pandas, torch, matplotlib موجود است.
# اجرا: python body_ml_analyzer_advanced.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # برای heatmap (اگر موجود نباشد، از matplotlib استفاده کنید)
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
from datetime import datetime
import json
import os
import logging

# تنظیم لاگینگ حرفه‌ای
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManualStandardScaler:
    """پیاده‌سازی دستی StandardScaler"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1  # جلوگیری از تقسیم بر صفر
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class ManualLabelEncoder:
    """پیاده‌سازی دستی LabelEncoder"""
    def __init__(self):
        self.classes_ = None
        self.mapping_ = None
    
    def fit(self, y):
        self.classes_ = np.unique(y)
        self.mapping_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        return np.array([self.mapping_.get(val, -1) for val in y])  # -1 برای ناشناخته
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        if self.classes_ is None:
            raise ValueError("Encoder not fitted")
        inverse_mapping = {idx: cls for cls, idx in self.mapping_.items()}
        return np.array([inverse_mapping.get(val, 'Unknown') for val in y])

def manual_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """پیاده‌سازی دستی train_test_split با stratify"""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    if stratify is not None:
        # Stratified split ساده
        unique_classes = np.unique(y)
        train_indices = []
        test_indices = []
        for cls in unique_classes:
            cls_indices = indices[y[indices] == cls]
            n_test = int(len(cls_indices) * test_size)
            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])
        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)
    else:
        n_test = int(len(indices) * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# تعریف مدل خارج از متد برای دسترسی جهانی
class BodyShapeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# پیاده‌سازی دستی confusion_matrix و classification_report
def manual_confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1
    return cm

def manual_classification_report(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    report = {}
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == label)
        report[label] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': support}
    return report

class BodyDataAnalyzer:
    """
    کلاس اصلی برای تجزیه و تحلیل داده‌های اندازه‌گیری بدن.
    شامل تولید داده، تجزیه و تحلیل، و آموزش مدل.
    """
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.df = None
        self.model = None
        self.scaler = ManualStandardScaler()
        self.label_encoder = ManualLabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
    
    def generate_synthetic_data(self):
        """
        تولید داده‌های مصنوعی بر اساس مثال‌های بسته Dart.
        ستون‌ها: bust, underbust, hip, waist, height, weight_kg, bicep, tricep, thigh, calf, body_shape (label)
        """
        logger.info("تولید داده‌های مصنوعی...")
        np.random.seed(42)  # برای تکرارپذیری
        
        # تولید داده‌های پایه - اطمینان از مثبت بودن با clip
        age = np.random.randint(18, 65, self.num_samples)
        gender_factor = np.random.choice([0.9, 1.1], self.num_samples)  # تفاوت جنسیتی ساده
        
        self.df = pd.DataFrame({
            'bust': np.clip(np.random.normal(36, 4, self.num_samples) * gender_factor, 20, 60),
            'underbust': np.clip(np.random.normal(32, 3, self.num_samples) * gender_factor, 20, 50),
            'hip': np.clip(np.random.normal(38.5, 5, self.num_samples) * gender_factor, 25, 60),
            'waist': np.clip(np.random.normal(28, 4, self.num_samples) * gender_factor, 20, 50),
            'height': np.clip(np.random.normal(65, 4, self.num_samples), 50, 80),
            'weight_kg': np.clip(np.random.normal(60, 10, self.num_samples), 40, 120),
            'bicep': np.clip(np.random.normal(13, 2, self.num_samples), 8, 20),
            'tricep': np.clip(np.random.normal(12.5, 1.5, self.num_samples), 8, 18),
            'thigh': np.clip(np.random.normal(22, 3, self.num_samples), 15, 35),
            'calf': np.clip(np.random.normal(14, 2, self.num_samples), 10, 20),
        })
        
        # محاسبه shape بدن بر اساس heuristic ساده (شبیه Dart)
        def determine_shape(row):
            bust_diff = abs(row['bust'] - row['waist'])
            hip_diff = abs(row['hip'] - row['waist'])
            if bust_diff < 2 and hip_diff < 2:
                return 'Rectangle'
            elif hip_diff > bust_diff + 2:
                return 'Pear'
            elif bust_diff > hip_diff + 2:
                return 'Inverted Triangle'
            elif row['waist'] < (row['bust'] * 0.75) and row['waist'] < (row['hip'] * 0.75):
                return 'Hourglass'
            else:
                return 'Apple'
        
        self.df['body_shape'] = self.df.apply(determine_shape, axis=1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.df.to_csv(f'synthetic_body_data_{timestamp}.csv', index=False)
        logger.info(f"داده‌ها در synthetic_body_data_{timestamp}.csv ذخیره شد. تعداد نمونه‌ها: {len(self.df)}")
        return self.df
    
    def data_analysis(self):
        """
        تجزیه و تحلیل داده‌ها: آمار توصیفی، همبستگی، و نمودارها - پیشرفته.
        """
        if self.df is None:
            self.generate_synthetic_data()
        
        logger.info("شروع تجزیه و تحلیل داده‌ها...")
        
        # آمار توصیفی
        print("\n📊 آمار توصیفی:")
        print(self.df.describe())
        
        # همبستگی با heatmap
        corr_matrix = self.df[['bust', 'hip', 'waist', 'height', 'weight_kg']].corr()
        print("\n🔗 ماتریس همبستگی:")
        print(corr_matrix)
        
        # Heatmap همبستگی - fallback اگر sns موجود نباشد
        plt.figure(figsize=(10, 8))
        if HAS_SNS:
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        else:
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
            plt.colorbar()
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center')
        plt.title('Heatmap همبستگی اندازه‌گیری‌های بدن')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # توزیع shape بدن
        shape_dist = self.df['body_shape'].value_counts()
        print("\n📈 توزیع شکل بدن:")
        print(shape_dist)
        
        # باکس‌پلات برای توزیع اندازه‌ها بر اساس shape
        plt.figure(figsize=(15, 10))
        self.df.boxplot(column=['bust', 'hip', 'waist'], by='body_shape', ax=plt.gca())
        plt.title('توزیع اندازه‌ها بر اساس شکل بدن')
        plt.suptitle('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('boxplot_by_shape.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # نمودارهای اصلی
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        self.df['bust'].hist(bins=20, ax=axes[0], alpha=0.7, color='skyblue')
        axes[0].set_title('توزیع اندازه سینه (Bust)')
        axes[0].set_xlabel('اندازه (اینچ)')
        
        self.df['hip'].hist(bins=20, ax=axes[1], alpha=0.7, color='lightgreen')
        axes[1].set_title('توزیع اندازه باسن (Hip)')
        axes[1].set_xlabel('اندازه (اینچ)')
        
        shape_dist.plot(kind='pie', ax=axes[2], autopct='%1.1f%%')
        axes[2].set_title('توزیع شکل بدن')
        
        plt.tight_layout()
        plt.savefig('advanced_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("نمودارهای پیشرفته در advanced_analysis_plots.png ذخیره شد.")
    
    def prepare_data_for_ml(self):
        """
        آماده‌سازی داده برای آموزش مدل: انکودینگ، اسکیلینگ، تقسیم داده - بدون sklearn.
        هدف: پیش‌بینی body_shape از ویژگی‌های اندازه‌گیری.
        """
        features = ['bust', 'underbust', 'hip', 'waist', 'height', 'weight_kg', 'bicep', 'tricep', 'thigh', 'calf']
        X = self.df[features].values
        y = self.label_encoder.fit_transform(self.df['body_shape'].values)
        
        # اسکیلینگ دستی
        X_scaled = self.scaler.fit_transform(X)
        
        # تقسیم داده دستی با stratify
        X_train, X_test, y_train, y_test = manual_train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # تبدیل به تنسور
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"داده‌ها آماده شد. کلاس‌ها: {self.label_encoder.classes_}")
        return self.train_loader, self.test_loader, X_test, y_test  # بازگشت y_test برای ارزیابی
    
    def define_model(self, input_size, num_classes):
        """
        تعریف مدل عصبی ساده با PyTorch - با لایه‌های بیشتر برای بهبود.
        """
        self.model = BodyShapeClassifier(input_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        logger.info("مدل پیشرفته تعریف شد.")
    
    def train_model(self, epochs=100):
        """
        آموزش مدل با منحنی یادگیری و ذخیره بهترین مدل.
        """
        if self.model is None:
            input_size = len(['bust', 'underbust', 'hip', 'waist', 'height', 'weight_kg', 'bicep', 'tricep', 'thigh', 'calf'])
            num_classes = len(self.label_encoder.classes_)
            self.define_model(input_size, num_classes)
        
        logger.info(f"شروع آموزش برای {epochs} эпох...")
        self.model.train()
        train_losses = []
        val_accuracies = []
        
        best_accuracy = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            train_losses.append(avg_loss)
            
            # ارزیابی ساده در هر epoch
            val_acc = self._validate_model()
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(self.model.state_dict(), 'best_body_shape_model.pth')
        
        # منحنی یادگیری
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('منحنی Loss آموزش')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('دقت اعتبارسنجی')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        torch.save(self.model.state_dict(), 'final_body_shape_model.pth')
        logger.info("مدل نهایی در final_body_shape_model.pth ذخیره شد. بهترین دقت: {:.2f}%".format(best_accuracy * 100))
    
    def _validate_model(self):
        """ارزیابی سریع در طول آموزش"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / total
    
    def evaluate_model(self, X_test=None, y_test=None):
        """
        ارزیابی پیشرفته: دقت، ماتریس درهم‌ریختگی، گزارش طبقه‌بندی.
        """
        if X_test is None or y_test is None:
            features = ['bust', 'underbust', 'hip', 'waist', 'height', 'weight_kg', 'bicep', 'tricep', 'thigh', 'calf']
            X = self.scaler.transform(self.df[features].values)
            y = self.label_encoder.transform(self.df['body_shape'].values)
            _, X_test, _, y_test = manual_train_test_split(X, y, test_size=0.2)
        
        logger.info("ارزیابی مدل...")
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            outputs = self.model(test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
            y_true = y_test
        
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        
        accuracy = np.mean(y_pred == y_true)
        print(f"\n🎯 دقت مدل: {accuracy * 100:.2f}%")
        
        # ماتریس درهم‌ریختگی
        cm = manual_confusion_matrix(y_true, y_pred, labels=np.arange(len(self.label_encoder.classes_)))
        plt.figure(figsize=(8, 6))
        if HAS_SNS:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        else:
            plt.imshow(cm, cmap='Blues', interpolation='none')
            plt.colorbar()
            plt.xticks(range(len(self.label_encoder.classes_)), self.label_encoder.classes_, rotation=45)
            plt.yticks(range(len(self.label_encoder.classes_)), self.label_encoder.classes_)
            for i in range(len(cm)):
                for j in range(len(cm)):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.title('ماتریس درهم‌ریختگی')
        plt.ylabel('واقعی')
        plt.xlabel('پیش‌بینی')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # گزارش طبقه‌بندی
        report = manual_classification_report(y_true, y_pred, labels=np.arange(len(self.label_encoder.classes_)))
        print("\n📋 گزارش طبقه‌بندی:")
        for label_idx, metrics in report.items():
            label_name = self.label_encoder.inverse_transform([int(label_idx)])[0]
            print(f"{label_name}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1-score={metrics['f1-score']:.2f}, support={metrics['support']}")
        
        logger.info(f"دقت: {accuracy * 100:.2f}%")
    
    def cross_validate(self, n_folds=5):
        """
        کراس-والیدیشن ساده برای ارزیابی مدل.
        """
        logger.info(f"کراس-والیدیشن با {n_folds} فولد...")
        features = ['bust', 'underbust', 'hip', 'waist', 'height', 'weight_kg', 'bicep', 'tricep', 'thigh', 'calf']
        X = self.scaler.fit_transform(self.df[features].values)
        y = self.label_encoder.fit_transform(self.df['body_shape'].values)
        
        fold_accuracies = []
        fold_size = len(X) // n_folds
        
        for i in range(n_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_folds - 1 else len(X)
            val_mask = np.arange(val_start, val_end)
            train_mask = np.concatenate((np.arange(0, val_start), np.arange(val_end, len(X))))
            
            X_train_fold, X_val_fold = X[train_mask], X[val_mask]
            y_train_fold, y_val_fold = y[train_mask], y[val_mask]
            
            # مدل موقت
            model_fold = BodyShapeClassifier(len(features), len(self.label_encoder.classes_)).to(self.device)
            criterion_fold = nn.CrossEntropyLoss()
            optimizer_fold = optim.Adam(model_fold.parameters(), lr=0.001)
            
            # آموزش سریع (10 epochs)
            model_fold.train()
            for _ in range(10):
                optimizer_fold.zero_grad()
                outputs = model_fold(torch.tensor(X_train_fold, dtype=torch.float32).to(self.device))
                loss = criterion_fold(outputs, torch.tensor(y_train_fold, dtype=torch.long).to(self.device))
                loss.backward()
                optimizer_fold.step()
            
            # ارزیابی
            model_fold.eval()
            with torch.no_grad():
                outputs = model_fold(torch.tensor(X_val_fold, dtype=torch.float32).to(self.device))
                _, predicted = torch.max(outputs, 1)
                acc = (predicted.cpu().numpy() == y_val_fold).mean()
                fold_accuracies.append(acc)
        
        cv_accuracy = np.mean(fold_accuracies)
        print(f"\n🔄 دقت میانگین کراس-والیدیشن: {cv_accuracy * 100:.2f}% (±{np.std(fold_accuracies) * 100:.2f}%)")
        logger.info(f"CV Accuracy: {cv_accuracy * 100:.2f}%")
    
    def predict_shape(self, measurements):
        """
        پیش‌بینی شکل بدن برای اندازه‌گیری‌های جدید.
        measurements: dict با کلیدهای ویژگی‌ها
        """
        if self.model is None:
            logger.warn("مدل آموزش ندیده است.")
            return None
        
        features = ['bust', 'underbust', 'hip', 'waist', 'height', 'weight_kg', 'bicep', 'tricep', 'thigh', 'calf']
        input_data = np.array([measurements.get(f, 0) for f in features]).reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            shape_idx = predicted.item()
            predicted_shape = self.label_encoder.inverse_transform([shape_idx])[0]
        
        return predicted_shape
    
    def interactive_predict(self):
        """
        پیش‌بینی تعاملی برای ورودی کاربر.
        """
        print("\n🖥️  پیش‌بینی تعاملی شکل بدن:")
        measurements = {}
        features = ['bust', 'underbust', 'hip', 'waist', 'height', 'weight_kg', 'bicep', 'tricep', 'thigh', 'calf']
        for f in features:
            try:
                val = float(input(f"وارد کنید {f} (اینچ/کیلو): "))
                measurements[f] = val
            except ValueError:
                logger.error(f"مقدار نامعتبر برای {f}")
                return None
        
        predicted = self.predict_shape(measurements)
        if predicted:
            print(f"🔮 شکل پیش‌بینی‌شده: {predicted}")
        return predicted

def main():
    analyzer = BodyDataAnalyzer(num_samples=2000)  # افزایش نمونه‌ها برای دقت بیشتر
    
    # 1. تولید و تجزیه و تحلیل داده
    df = analyzer.generate_synthetic_data()
    analyzer.data_analysis()
    
    # 2. آماده‌سازی داده برای ML
    train_loader, test_loader, X_test, y_test = analyzer.prepare_data_for_ml()
    
    # 3. آموزش مدل
    analyzer.train_model(epochs=100)  # epochs بیشتر
    
    # 4. ارزیابی پیشرفته
    analyzer.evaluate_model(X_test, y_test)
    
    # 5. کراس-والیدیشن
    analyzer.cross_validate(n_folds=5)
    
    # 6. پیش‌بینی مثال (بر اساس Dart)
    example_measurements = {
        'bust': 36.0, 'underbust': 32.0, 'hip': 38.5, 'waist': 28.0,
        'height': 65.0, 'weight_kg': 60.0, 'bicep': 13.0, 'tricep': 12.5,
        'thigh': 22.0, 'calf': 14.0
    }
    predicted = analyzer.predict_shape(example_measurements)
    print(f"\n🔮 پیش‌بینی شکل بدن برای مثال: {predicted}")
    
    # 7. پیش‌بینی تعاملی (اختیاری)
    if input("آیا می‌خواهید پیش‌بینی تعاملی انجام دهید؟ (y/n): ").lower() == 'y':
        analyzer.interactive_predict()
    
    # ذخیره خلاصه پیشرفته
    summary = {
        'num_samples': analyzer.num_samples,
        'classes': list(analyzer.label_encoder.classes_),
        'timestamp': datetime.now().isoformat(),
        'model_path': 'final_body_shape_model.pth',
        'data_path': 'synthetic_body_data_*.csv'
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'ml_summary_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    logger.info(f"خلاصه پیشرفته در ml_summary_{timestamp}.json ذخیره شد.")

if __name__ == "__main__":
    main()
