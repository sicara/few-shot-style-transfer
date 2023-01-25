import pandas as pd
import matplotlib.pyplot as plt


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
):
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
        * 100,
        len(query_results_same_color_as_class_representative),
    )


def compute_accuracy_for_samples_with_same_color_as_other_class_representative(
    query_results: pd.DataFrame,
):
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
        * 100,
        len(query_results_same_color_as_other_class_representative),
    )


def compute_accuracy_for_samples_with_same_color_as_no_class_representative(
    query_results: pd.DataFrame,
):
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
        * 100,
        len(query_results_same_color_as_no_class_representative),
    )


def compute_task_color_similarity(query_results: pd.DataFrame, task_id: int) -> float:
    """Compute a color similarity index of a task in percent. It adds to the mean 50%,
    the percentage of samples with the same color as their class representative;
    and subtracts the percentage of samples with the same color as another class representative.
    Higher the index is, more the samples are similar to their class representative.
    Args:
        query_results: the dataframe gathering the results of the classification
        task_id: the task ID for which we want to compute the index
    """
    task_query_results = query_results[(query_results["task_id"] == task_id)]
    return (
        0.5
        + len(
            pd.concat(
                [
                    task_query_results[
                        (task_query_results["true_label"] == 1)
                        & (
                            task_query_results["color"]
                            == task_query_results["support_set_1_color"]
                        )
                    ],
                    task_query_results[
                        (task_query_results["true_label"] == 0)
                        & (
                            task_query_results["color"]
                            == task_query_results["support_set_0_color"]
                        )
                    ],
                ]
            )
        )
        / len(task_query_results)
        - len(
            pd.concat(
                [
                    task_query_results[
                        (task_query_results["true_label"] == 1)
                        & (
                            task_query_results["color"]
                            == task_query_results["support_set_0_color"]
                        )
                    ],
                    task_query_results[
                        (task_query_results["true_label"] == 0)
                        & (
                            task_query_results["color"]
                            == task_query_results["support_set_1_color"]
                        )
                    ],
                ]
            )
        )
        / len(task_query_results)
    ) * 100


def plot_task_accuracy_and_color_similarity(query_results: pd.DataFrame):
    number_of_tasks = len(list(query_results["task_id"].unique()))
    task_accuracy_list = [
        compute_accuracy_for_one_task(query_results, task_id)
        for task_id in range(number_of_tasks)
    ]
    task_color_similarity_list = [
        compute_task_color_similarity(query_results, task_id)
        for task_id in range(number_of_tasks)
    ]
    metrics_per_task = (
        pd.DataFrame(
            list(
                zip(
                    [i for i in range(number_of_tasks)],
                    task_accuracy_list,
                    task_color_similarity_list,
                )
            ),
            columns=["task_id", "accuracy", "color_similarity"],
        )
        .sort_values(by=["accuracy"], ascending=True)
        .reset_index(drop=True)
    )
    metrics_per_task = metrics_per_task.assign(
        smooth_color_similarity=lambda df: df.color_similarity.rolling(
            window=100
        ).mean()
    ).dropna()
    fig, ax = plt.subplots()
    ax.plot(metrics_per_task["accuracy"], color="red", marker="o")
    ax.set_xlabel("task_id", fontsize=10)
    ax.set_ylabel("accuracy", color="red", fontsize=10)
    ax.set_ylim((0, 100))
    ax2 = ax.twinx()
    ax2.plot(metrics_per_task["smooth_color_similarity"], color="blue", marker="o")
    ax2.set_ylabel("color similarity index", color="blue", fontsize=10)
    plt.show()


def color_representativeness_index(query_results: pd.DataFrame, task_id: int) -> float:
    task_query_results = query_results[(query_results["task_id"] == task_id)]
    return len(
        pd.concat(
            [
                task_query_results[
                    (task_query_results["true_label"] == 1)
                    & (
                        task_query_results["color"]
                        == task_query_results["support_set_1_color"]
                    )
                ],
                task_query_results[
                    (task_query_results["true_label"] == 0)
                    & (
                        task_query_results["color"]
                        == task_query_results["support_set_0_color"]
                    )
                ],
            ]
        )
    ) / len(task_query_results)


def color_difference_index(query_results: pd.DataFrame, task_id: int) -> float:
    task_query_results = query_results[(query_results["task_id"] == task_id)]
    return len(
        pd.concat(
            [
                task_query_results[
                    (task_query_results["true_label"] == 1)
                    & (
                        task_query_results["color"]
                        == task_query_results["support_set_0_color"]
                    )
                ],
                task_query_results[
                    (task_query_results["true_label"] == 0)
                    & (
                        task_query_results["color"]
                        == task_query_results["support_set_1_color"]
                    )
                ],
            ]
        )
    ) / len(task_query_results)


def plot_task_accuracy_and_color_indexes(query_results: pd.DataFrame):
    number_of_tasks = len(list(query_results["task_id"].unique()))
    task_accuracy_list = [
        compute_accuracy_for_one_task(query_results, task_id)
        for task_id in range(number_of_tasks)
    ]
    task_color_representativeness_list = [
        color_representativeness_index(query_results, task_id)
        for task_id in range(number_of_tasks)
    ]
    task_color_difference_list = [
        color_difference_index(query_results, task_id)
        for task_id in range(number_of_tasks)
    ]
    metrics_per_task = (
        pd.DataFrame(
            list(
                zip(
                    [i for i in range(number_of_tasks)],
                    task_accuracy_list,
                    task_color_representativeness_list,
                    task_color_difference_list,
                )
            ),
            columns=["task_id", "accuracy", "color_similarity", "color_difference"],
        )
        .sort_values(by=["accuracy"], ascending=True)
        .reset_index(drop=True)
    )
    metrics_per_task = metrics_per_task.assign(
        smooth_color_similarity=lambda df: df.color_similarity.rolling(
            window=100
        ).mean(),
        smooth_color_difference=lambda df: df.color_difference.rolling(
            window=100
        ).mean(),
    ).dropna()
    fig, ax = plt.subplots()
    ax.plot(metrics_per_task["accuracy"], color="red", marker="o")
    ax.set_xlabel("task_id", fontsize=10)
    ax.set_ylabel("accuracy", color="red", fontsize=10)
    ax.set_ylim((0, 100))
    ax2 = ax.twinx()
    ax2.plot(
        metrics_per_task["smooth_color_similarity"],
        color="blue",
        marker="o",
        markersize=3,
    )
    ax2.plot(
        metrics_per_task["smooth_color_difference"],
        color="green",
        marker="o",
        markersize=3,
    )
    ax2.legend(["similarity index", "difference index"])
    ax2.set_ylabel("color indexes", color="black", fontsize=10)
    plt.show()
