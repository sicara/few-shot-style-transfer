import pandas as pd


def compute_total_accuracy(query_results: pd.DataFrame) -> float:
    return (
        len(
            query_results[
                query_results["true_label"] == query_results["predicted_label"]
            ]
        )
        / len(query_results)
        * 100
    )


def compute_accuracy_for_one_task(query_results: pd.DataFrame, task_id: int) -> float:
    return (
        len(
            query_results[
                (query_results["task_id"] == task_id)
                & (query_results["true_label"] == query_results["predicted_label"])
            ]
        )
        / len(query_results[(query_results["task_id"] == task_id)])
        * 100
    )


def compute_accuracy_for_samples_with_same_color_as_class_representative(
    query_results: pd.DataFrame,
) -> float:
    query_results_same_color_as_class_representative = pd.concat(
        [
            query_results[
                (query_results["true_label"] == 1)
                & (query_results["color"] == query_results["support_set_1_color"])
            ],
            query_results[
                (query_results["true_label"] == 0)
                & (query_results["color"] == query_results["support_set_0_color"])
            ],
        ]
    )
    return (
        len(
            query_results_same_color_as_class_representative[
                query_results_same_color_as_class_representative["true_label"]
                == query_results_same_color_as_class_representative["predicted_label"]
            ]
        )
        / len(query_results_same_color_as_class_representative)
        * 100
    )


def compute_accuracy_for_samples_with_same_color_as_other_class_representative(
    query_results: pd.DataFrame,
) -> float:
    query_results_same_color_as_other_class_representative = pd.concat(
        [
            query_results[
                (query_results["true_label"] == 1)
                & (query_results["color"] == query_results["support_set_0_color"])
            ],
            query_results[
                (query_results["true_label"] == 0)
                & (query_results["color"] == query_results["support_set_1_color"])
            ],
        ]
    )
    return (
        len(
            query_results_same_color_as_other_class_representative[
                query_results_same_color_as_other_class_representative["true_label"]
                == query_results_same_color_as_other_class_representative[
                    "predicted_label"
                ]
            ]
        )
        / len(query_results_same_color_as_other_class_representative)
        * 100
    )


def compute_accuracy_for_samples_with_same_color_as_no_class_representative(
    query_results: pd.DataFrame,
) -> float:
    query_results_same_color_as_no_class_representative = query_results[
        (query_results["color"] != query_results["support_set_0_color"])
        & (query_results["color"] != query_results["support_set_1_color"])
    ]
    return (
        len(
            query_results_same_color_as_no_class_representative[
                query_results_same_color_as_no_class_representative["true_label"]
                == query_results_same_color_as_no_class_representative[
                    "predicted_label"
                ]
            ]
        )
        / len(query_results_same_color_as_no_class_representative)
        * 100
    )
