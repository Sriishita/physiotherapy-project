def recommend_exercise(problem, sub_problem=None):
    if problem == "shoulder":
        if sub_problem == "mobility":
            return "lateral_raise"
        return "arm_raise"

    elif problem == "knee":
        return "squat"

    elif problem == "general":
        return "leg_raise"

    return "arm_raise"