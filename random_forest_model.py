import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# ===============================
# SAFE INPUT FUNCTIONS
# ===============================
def ask_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter an integer.")


def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")


# ===============================
# ATTRACTIVE RESULT CARD (GRAPH)
# ===============================
def show_prediction_card(probability, prediction):
    """
    probability = [p_not, p_yes]
    prediction = 0 or 1
    """
    p_not = float(probability[0])
    p_yes = float(probability[1])

    verdict = "NOT LIKELY TO PROCRASTINATE" if prediction == 0 else "LIKELY TO PROCRASTINATE"

    plt.figure(figsize=(7, 4))
    plt.bar(["Not Procrastinator", "Procrastinator"], [p_not, p_yes])
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Student Behavioral Prediction")

    # Big verdict text
    plt.text(
        0.5, 0.92, verdict,
        ha="center", va="center",
        transform=plt.gca().transAxes,
        fontsize=14, fontweight="bold"
    )

    # Values above bars
    plt.text(0, p_not + 0.03, f"{p_not:.2f}", ha="center")
    plt.text(1, p_yes + 0.03, f"{p_yes:.2f}", ha="center")

    plt.tight_layout()
    plt.show()


# ===============================
# OPTIONAL: TOP FACTORS CARD (GRAPH)
# ===============================
def show_top_factors_graph(feat_df, new_student, top_n=5):
    top = feat_df.head(top_n).copy()
    top = top.sort_values("Importance", ascending=True)  # for nicer barh

    labels = []
    values = []
    for f in top["Feature"].tolist():
        labels.append(f)
        values.append(float(new_student.get(f, 0)))

    plt.figure(figsize=(9, 5))
    plt.barh(labels, values)
    plt.xlabel("Entered value (student)")
    plt.title(f"Top {top_n} Influencing Features (Student Input Values)")
    plt.tight_layout()
    plt.show()


# ===============================
# MAIN PROGRAM
# ===============================
def main():
    # 1) LOAD DATA
    data = pd.read_csv("student_academic_cleaned.csv")
    data.columns = data.columns.str.strip()  # fixes hidden \n in column names

    # 2) CHECK LABEL
    if "Procrastination_Label" not in data.columns:
        raise ValueError(
            "Procrastination_Label column not found in student_academic_cleaned.csv. "
            "Create it in your cleaning/label step first."
        )

    # 3) FEATURES & TARGET
    X = data.drop("Procrastination_Label", axis=1)
    y = data["Procrastination_Label"]

    print("\nLabel distribution:")
    print(y.value_counts())

    # 4) TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) TRAIN MODEL
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6) EVALUATION
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Accuracy:")
    print(f"Accuracy = {accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Prepare ROC values (we will DISPLAY after student input)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Feature importance table
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importances:")
    print(feat_df)

    # ===============================
    # 7) INTERACTIVE STUDENT INPUT
    # ===============================
    print("\n==============================")
    print("REAL-TIME STUDENT PREDICTION")
    print("==============================")

    # Ask once for smartphone hours; derive cleaned automatically
    age = ask_int("Enter Age: ")
    gender = ask_int("Enter Gender (0 = Female, 1 = Male): ")
    study_hours = ask_int("Study hours per week: ")
    study_days = ask_int("Study days per week: ")
    smartphone_hours = ask_int("Daily smartphone hours: ")
    stress = ask_int("Stress level (1-5): ")
    delay = ask_int("Delay level (1-5): ")
    gpa = ask_float("Enter GPA / Score: ")

    # Build student record (keys must match your CSV columns)
    new_student = {
        "Age": age,
        "Gender": gender,
        "How many hours do you study per week on average?": study_hours,
        "On how many days per week do you study consistently?": study_days,
        "How much time do you spend on smartphone/social media daily?": smartphone_hours,
        "Smartphone_Time_Cleaned": smartphone_hours,  # derived
        "I feel stressed about my studies.": stress,
        "I often delay starting assignments or studying until the last moment.": delay,
        "Your current GPA / Academic performance": gpa,
        "Q": 0
    }

    # Align columns with training data; missing columns become 0
    new_df = pd.DataFrame([new_student]).reindex(columns=X.columns, fill_value=0)

    # Predict
    prediction = model.predict(new_df)[0]
    probability = model.predict_proba(new_df)[0]

    # Print result in terminal
    print("\nPrediction Result:")
    if prediction == 1:
        print("Student is likely a PROCRASTINATOR")
    else:
        print("Student is NOT likely a procrastinator")

    print("\nPrediction Probabilities [Not Procrastinator, Procrastinator]:")
    print(probability)

    print("\nTop 5 influencing features (with entered values):")
    for feature in feat_df.head(5)["Feature"]:
        print(f"{feature}: {new_student.get(feature, 0)}")

    # ===============================
    # 8) ATTRACTIVE VISUAL OUTPUT
    # ===============================
    # Prediction card (probability bars + verdict text)
    show_prediction_card(probability, prediction)

    # Optional: show top factors as a graph of entered values
    # (This is student-specific "behavioral analysis" visualization)
    show_top_factors_graph(feat_df, new_student, top_n=5)

    # ===============================
    # 9) ROC CURVE (MODEL PERFORMANCE)
    # ===============================
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Procrastination Prediction (Model Performance)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
