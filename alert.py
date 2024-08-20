import scipy.io
import tkinter as tk
from tkinter import messagebox

# Load the .mat file
data = scipy.io.loadmat('eeg_results_improved.mat')
y_test = data['y_test'].flatten()
y_pred = data['y_pred'].flatten()

# Set a threshold for alerts
threshold = 0.5  # Adjust based on your model's output

# Initialize the main window
root = tk.Tk()
root.title("Epilepsy Detection Alert System")

alerted_indices = []

def check_alerts():
    global alerted_indices
    new_alerts = False

    # Check predictions against the threshold
    for i, pred in enumerate(y_pred):
        if pred > threshold and i not in alerted_indices:
            alerted_indices.append(i)
            new_alerts = True

    # Summarize alerts if any new ones were found
    if new_alerts:
        alert_message = f"New potential epileptic events detected at indices: {alerted_indices[-5:]}"
        if len(alerted_indices) > 5:
            alert_message += f"\n...and {len(alerted_indices) - 5} more."
        messagebox.showwarning("Alert", alert_message)

    # Schedule the function to run again after 1000 ms (1 second)
    root.after(1000, check_alerts)

# Start the first check
check_alerts()

# Run the GUI loop
root.mainloop()
