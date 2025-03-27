from typing import List, Set
import pandas as pd


def exact_parent_duration(data: pd.DataFrame, method):
    if method == "merge":
        return _cal_exact_merge(data)
    elif method == "max":
        return _cal_exact_max(data)


def _cal_exact_merge(data: pd.DataFrame):
    def groupByParentLevel(potentialConflicGrp: pd.DataFrame):
        conflictionGrp: List[List[int]] = []
        potentialConflicGrp.apply(
            findAllConflict,
            axis=1,
            conflictionGrp=conflictionGrp,
            potentialConflicGrp=potentialConflicGrp,
        )

        childDuration = 0
        for grp in conflictionGrp:
            conflictChildren = potentialConflicGrp.loc[list(grp)]
            childDuration += (
                conflictChildren["endTime"].max() - conflictChildren["startTime"].min()
            )

        potentialConflicGrp = potentialConflicGrp.assign(
            exactParentDuration=potentialConflicGrp["parentDuration"] - childDuration
        )

        return potentialConflicGrp

    def findAllConflict(
        span, conflictionGrp: List[Set[int]], potentialConflicGrp: pd.DataFrame
    ):
        myStart = span["startTime"]
        myEnd = span["endTime"]
        """
        Three different types of confliction
                -------------------
                |     ThisSpan    |
                -------------------
        ---------------------
        |     OtherSpan     |
        ---------------------
        """
        conditionOne = (potentialConflicGrp["startTime"] < myStart) & (
            myStart < potentialConflicGrp["endTime"]
        )
        """
        ------------------
        |    ThisSpan    |
        ------------------
                ---------------------
                |     OtherSpan     |
                ---------------------
        """
        conditionTwo = (potentialConflicGrp["startTime"] < myEnd) & (
            myEnd < potentialConflicGrp["endTime"]
        )
        """
        ------------------------------
        |          ThisSpan          |
        ------------------------------
            ---------------------
            |     OtherSpan     |
            ---------------------
        """
        conditionThree = (myStart < potentialConflicGrp["startTime"]) & (
            myEnd > potentialConflicGrp["endTime"]
        )
        confliction = potentialConflicGrp.loc[
            conditionOne | conditionTwo | conditionThree
        ].index.to_list()
        confliction.append(span.name)
        correspondingGroup = set()
        for group in conflictionGrp:
            founded = False
            for index in confliction:
                if index in group:
                    correspondingGroup = group
                    founded = True
                    break
            if founded:
                break
        for index in confliction:
            correspondingGroup.add(index)
        if not correspondingGroup in conflictionGrp:
            conflictionGrp.append(correspondingGroup)

    data = data.groupby("traceId").apply(
        lambda x: x.groupby("parentId").apply(groupByParentLevel)
    )

    data = (
        data.drop(columns=["traceId", "parentId"]).reset_index().drop(columns="level_2")
    )

    data = data.astype({"exactParentDuration": float, "childDuration": float})
    data = data.loc[data["exactParentDuration"] > 0]

    return data


def _cal_exact_max(data: pd.DataFrame):
    data = data.groupby("traceId").apply(
        lambda x: x.groupby("parentId").apply(
            lambda y: y.assign(
                exactParentDuration=y["parentDuration"] - y["childDuration"].max()
            )
        )
    )
    return (
        data.drop(columns=["traceId", "parentId"]).reset_index().drop(columns="level_2")
    )


def decouple_parent_and_child(data: pd.DataFrame, percentile=0.95):
    parent_perspective = data.groupby(["parentMS", "parentPod"])[
        "exactParentDuration"
    ].quantile(percentile)
    parent_perspective.index.names = ["microservice", "pod"]
    child_perspective = data.groupby(["childMS", "childPod"])["childDuration"].quantile(
        percentile
    )
    child_perspective.index.names = ["microservice", "pod"]
    quantiled = pd.concat([parent_perspective, child_perspective])
    quantiled = quantiled[~quantiled.index.duplicated(keep="first")]
    # Parse the serie to a data frame
    data = quantiled.to_frame(name="latency")
    data = data.reset_index()
    return data


def construct_relationship(data: pd.DataFrame, max_edges):
    result = pd.DataFrame()

    def graph_for_trace(x: pd.DataFrame):
        nonlocal max_edges
        if len(x) <= max_edges:
            return
        max_edges = len(x)
        root = x.loc[~x["parentId"].isin(x["childId"].unique().tolist())]

        nonlocal result
        result = pd.DataFrame()

        def dfs(parent, parent_tag):
            nonlocal result
            children = x.loc[x["parentId"] == parent["childId"]]
            for index, (_, child) in enumerate(children.iterrows()):
                child_tag = f"{parent_tag}.{index+1}"
                result = pd.concat(
                    [result, pd.DataFrame(child).transpose().assign(tag=child_tag)]
                )
                dfs(child, child_tag)

        for index, (_, first_lvl) in enumerate(root.iterrows()):
            first_lvl["tag"] = index + 1
            result = pd.concat([result, pd.DataFrame(first_lvl).transpose()])
            dfs(first_lvl, index + 1)

    data.groupby(["traceId", "traceTime"]).apply(graph_for_trace)
    if len(result) != 0:
        return (
            result[
                [
                    "parentMS",
                    "childMS",
                    "parentOperation",
                    "childOperation",
                    "tag",
                    "service",
                ]
            ],
            max_edges,
        )
    else:
        return False


def remove_repeatation(data: pd.DataFrame):
    roots = data.loc[
        (~data["parentId"].isin(data["childId"]))
        & (data["parentMS"] == data["childMS"])
    ]

    def remove_prev_level_child(prev_level):
        nonlocal data
        # Find all next level
        repeat_children = data.loc[
            (data["parentId"] == prev_level["childId"])
            & (data["parentMS"] == data["childMS"])
        ]
        other_children = data.loc[
            (data["parentId"] == prev_level["childId"])
            & ~(data["parentMS"] == data["childMS"])
        ]
        # Remove repeat for next level's repeat children
        if len(repeat_children) != 0:
            for _, child in repeat_children.iterrows():
                next_level_children = remove_prev_level_child(child)
                other_children = pd.concat([other_children, next_level_children])
        # Build relationship between previous level's parent and next level's children
        other_children["parentId"] = prev_level["parentId"]
        other_children["parentDuration"] = prev_level["parentDuration"]
        other_children["parentOperation"] = prev_level["parentOperation"]
        # Remove previous level
        data = data.loc[~(data["childId"] == prev_level["childId"])]
        data = data.loc[~data["childId"].isin(other_children["childId"])]
        data = pd.concat([data, other_children])
        return other_children

    for _, root in roots.iterrows():
        remove_prev_level_child(root)
    return data


def no_entrance_trace_duration(spans_data: pd.DataFrame, entrance_name):
    roots = spans_data[
        (spans_data["parentMS"].str.contains(entrance_name))
        & (~spans_data["childMS"].str.contains(entrance_name))
    ]
    roots = roots.assign(endTime=roots["startTime"] + roots["childDuration"])
    start_time = roots.groupby(["traceTime", "traceId"], sort=False)["startTime"].min()
    end_time = roots.groupby(["traceTime", "traceId"], sort=False)["endTime"].max()
    return (
        roots.drop_duplicates("traceId")
        .assign(traceDuration=(end_time - start_time).values)[
            ["traceId", "traceTime", "traceDuration"]
        ]
        .rename(columns={"traceDuration": "traceLatency"})
    )
    
def calculate_parallel_fanout(df: pd.DataFrame, time_threshold_us: int = 1000):
    """
    하나의 trace 내에서, 부모 span이 동시에 호출한 자식 span의 수 (병렬 fan-out) 계산
    time_threshold_us: 병렬로 간주할 시간 간격 (마이크로초 단위)
    """
    max_parallel = 0

    for parent_id, group in df.groupby("parentId"):
        if len(group) <= 1:
            continue

        # 자식 span의 시작/종료 시각으로 정렬
        spans = group[["startTime", "endTime"]].sort_values("startTime").to_numpy()

        parallel_count = 1
        for i in range(len(spans)):
            current_end = spans[i][1]
            count = 1
            for j in range(i + 1, len(spans)):
                next_start = spans[j][0]
                if next_start - current_end <= time_threshold_us:
                    count += 1
                else:
                    break
            parallel_count = max(parallel_count, count)

        max_parallel = max(max_parallel, parallel_count)

    return max_parallel
    
def is_complex_trace(df: pd.DataFrame, ms_threshold):
    num_services = len(set(df["childMS"]).union(set(df["parentMS"])))
    #fan_out = calculate_parallel_fanout(df)
    #fan_out = df["parentId"].value_counts().max()
    #print("fan out:", fan_out)
    return num_services >= ms_threshold 