import pandas as pd
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score
)

class Logistic_Regression_binomial:
    def __init__(self, root):
        self.root = root
        self.root.title("Logistic Regression Binomial")
        self.root.geometry("600x700")  # Adjusted window size for additional inputs
        self.root.resizable(False, False)

        # Set the appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Create UI components
        self.create_widgets()

    def create_widgets(self):
        # Create a main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Create and place entry fields and labels for user input
        self.test_size_label = ctk.CTkLabel(self.main_frame, text="Test Size (0-1):")
        self.test_size_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.test_size_entry = ctk.CTkEntry(self.main_frame)
        self.test_size_entry.grid(row=0, column=1, padx=10, pady=5)

        self.random_state_label = ctk.CTkLabel(self.main_frame, text="Random State:")
        self.random_state_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.random_state_entry = ctk.CTkEntry(self.main_frame)
        self.random_state_entry.grid(row=1, column=1, padx=10, pady=5)

        # Dropdown for graph selection
        self.graph_type_label = ctk.CTkLabel(self.main_frame, text="Select Graph:")
        self.graph_type_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.graph_type_var = ctk.StringVar(value="Bar Plot")
        self.graph_type_menu = ctk.CTkOptionMenu(
            self.main_frame, variable=self.graph_type_var,
            values=["Bar Plot", "Boxplot", "Bar Chart"]
        )
        self.graph_type_menu.grid(row=2, column=1, padx=10, pady=5)

        # Dropdowns for column selection
        self.x_col_label = ctk.CTkLabel(self.main_frame, text="Select X Column:")
        self.x_col_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
        self.x_col_var = ctk.StringVar(value="")
        self.x_col_menu = ctk.CTkOptionMenu(self.main_frame, variable=self.x_col_var)
        self.x_col_menu.grid(row=3, column=1, padx=10, pady=5)

        self.y_col_label = ctk.CTkLabel(self.main_frame, text="Select Y Column:")
        self.y_col_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.y_col_var = ctk.StringVar(value="")
        self.y_col_menu = ctk.CTkOptionMenu(self.main_frame, variable=self.y_col_var)
        self.y_col_menu.grid(row=4, column=1, padx=10, pady=5)

        # Create and place buttons using grid layout for responsiveness
        self.load_button = ctk.CTkButton(self.main_frame, text="Load Data", command=self.load_data, border_width=2, corner_radius=8)
        self.load_button.grid(row=5, column=0, padx=10, pady=5, sticky='ew', columnspan=2)

        self.train_button = ctk.CTkButton(self.main_frame, text="Train Model", command=self.train_model, border_width=2, corner_radius=8)
        self.train_button.grid(row=6, column=0, padx=10, pady=5, sticky='ew', columnspan=2)

        self.metrics_button = ctk.CTkButton(self.main_frame, text="Show Metrics", command=self.show_metrics, border_width=2, corner_radius=8)
        self.metrics_button.grid(row=7, column=0, padx=10, pady=5, sticky='ew', columnspan=2)

        self.roc_button = ctk.CTkButton(self.main_frame, text="Show ROC Curve", command=self.show_roc_curve, border_width=2, corner_radius=8)
        self.roc_button.grid(row=8, column=0, padx=10, pady=5, sticky='ew', columnspan=2)

        self.heatmap_button = ctk.CTkButton(self.main_frame, text="Show Correlation Heatmap", command=self.show_heatmap, border_width=2, corner_radius=8)
        self.heatmap_button.grid(row=9, column=0, padx=10, pady=5, sticky='ew', columnspan=2)

        self.visualize_button = ctk.CTkButton(self.main_frame, text="Show Selected Visualization", command=self.show_selected_visualization, border_width=2, corner_radius=8)
        self.visualize_button.grid(row=10, column=0, padx=10, pady=5, sticky='ew', columnspan=2)

        # Configure grid to expand with window resizing
        for row in range(11):
            self.main_frame.grid_rowconfigure(row, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.update_column_options()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")

    def update_column_options(self):
        columns = self.df.columns.tolist()
        self.x_col_menu.configure(values=columns)
        self.y_col_menu.configure(values=columns)

    def train_model(self):
        if not hasattr(self, 'df'):
            messagebox.showerror("Error", "Please load the data first.")
            return

        # Validate user input
        try:
            test_size = float(self.test_size_entry.get())
            random_state = int(self.random_state_entry.get())
            if not (0 < test_size < 1):
                raise ValueError("Test size must be between 0 and 1.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        # Prepare the data
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        # Split the data into training and testing sets
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the Logistic Regression model
        self.model = LogisticRegression()
        self.model.fit(X_train, self.y_train)

        # Prediction
        self.y_pred = self.model.predict(X_test)

        self.acc = accuracy_score(self.y_test, self.y_pred)
        self.y_prob = self.model.predict_proba(X_test)[:, 1]
        self.fpr, self.tpr, _ = roc_curve(self.y_test, self.y_prob)
        self.roc_auc = auc(self.fpr, self.tpr)

        messagebox.showinfo("Training Complete", f"Model trained with accuracy: {self.acc*100:.2f}%")

    def show_metrics(self):
        if not hasattr(self, 'y_pred'):
            messagebox.showerror("Error", "Please train the model first.")
            return

        # Calculate precision, recall, and f1-score
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')

        metrics_text = (
            f"Accuracy: {self.acc*100:.2f}%\n"
            f"Precision: {precision:.2f}\n"
            f"Recall: {recall:.2f}\n"
            f"F1-score: {f1:.2f}\n\n"
            f"Confusion Matrix:\n{confusion_matrix(self.y_test, self.y_pred)}\n\n"
            f"Classification Report:\n{classification_report(self.y_test, self.y_pred)}"
        )
        messagebox.showinfo("Model Metrics", metrics_text)

    def show_roc_curve(self):
        if not hasattr(self, 'y_pred'):
            messagebox.showerror("Error", "Please train the model first.")
            return

        plt.figure(figsize=(8, 6))
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {self.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def show_heatmap(self):
        if not hasattr(self, 'df'):
            messagebox.showerror("Error", "Please load the data first.")
            return

        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap')
        plt.show()

    def show_selected_visualization(self):
        if not hasattr(self, 'df'):
            messagebox.showerror("Error", "Please load the data first.")
            return

        selected_graph = self.graph_type_var.get()
        x_col = self.x_col_var.get()
        y_col = self.y_col_var.get()

        if selected_graph == "Bar Plot":
            if x_col and y_col:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=x_col, y=y_col, data=self.df)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'Bar Plot of {x_col} vs {y_col}')
                plt.show()
            else:
                messagebox.showerror("Input Error", "Please select both X and Y columns.")
        
        elif selected_graph == "Boxplot":
            if x_col and y_col:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=x_col, y=y_col, data=self.df)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'Boxplot of {x_col} by {y_col}')
                plt.show()
            else:
                messagebox.showerror("Input Error", "Please select both X and Y columns.")

        elif selected_graph == "Bar Chart":
            if x_col:
                plt.figure(figsize=(8, 6))
                self.df[x_col].value_counts().plot(kind='bar')
                plt.xlabel(x_col)
                plt.ylabel('Count')
                plt.title(f'Bar Chart of {x_col}')
                plt.show()
            else:
                messagebox.showerror("Input Error", "Please select a column for the bar chart.")

if __name__ == "__main__":
    root = ctk.CTk()
    app = Logistic_Regression_binomial(root)
    root.mainloop()
