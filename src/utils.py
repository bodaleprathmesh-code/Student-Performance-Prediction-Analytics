"""
Utility Functions
Student Performance Prediction Project
"""

# ----------------------------------------------------
# Performance Category
# ----------------------------------------------------

def get_performance_category(marks):
    """
    Returns performance category based on predicted marks.
    """

    if marks >= 90:
        return "Excellent"

    elif marks >= 75:
        return "Good"

    else:
        return "Need Improvement"


# ----------------------------------------------------
# Academic Risk Level
# ----------------------------------------------------

def get_risk_level(marks):
    """
    Returns academic risk level.
    """

    if marks >= 90:
        return "Very Low Risk"

    elif marks >= 75:
        return "Low Risk"

    elif marks >= 60:
        return "Moderate Risk"

    else:
        return "High Risk"


# ----------------------------------------------------
# Personalized Recommendations
# ----------------------------------------------------

def get_recommendations(marks):
    """
    Returns personalized recommendations.
    """

    recommendations = []

    if marks < 75:

        recommendations.append(
            "Increase daily study hours by 1–2 hours."
        )

        recommendations.append(
            "Improve attendance above 90%."
        )

        recommendations.append(
            "Practice quizzes regularly."
        )

        recommendations.append(
            "Focus on internal assessments."
        )

        recommendations.append(
            "Revise previous topics every week."
        )

    elif marks < 90:

        recommendations.append(
            "Maintain your current study routine."
        )

        recommendations.append(
            "Improve quiz performance."
        )

        recommendations.append(
            "Aim for higher internal marks."
        )

        recommendations.append(
            "Continue regular revision."
        )

    else:

        recommendations.append(
            "Excellent performance! Keep it up."
        )

        recommendations.append(
            "Help classmates through group study."
        )

        recommendations.append(
            "Practice advanced concepts."
        )

    return recommendations