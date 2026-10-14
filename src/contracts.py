"""Scientific constants shared by training, inference and reports.

These values describe the existing checkpoint, not user preferences. Changing
them requires scientific validation rather than a routine software cleanup.
"""

IMAGE_SIZE = 224
LABELS = ("NORMAL", "PNEUMONIA")  # Logit positions are part of the saved model's meaning.
