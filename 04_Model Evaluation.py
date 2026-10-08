# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy:",accuracy)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Stay (0)", "Churn (1)"]
    )
)

from sklearn.metrics import confusion_matrix

# Confusion Matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

