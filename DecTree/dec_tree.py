import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    The Gini criterion here means the following function:
        $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
        $R$ is a set of objects, $R_l$ and $R_r$ are objects that fall into the left and right subtree,
        $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — the proportion of objects of class 1 and 0, respectively.

        Instructions:
        * Thresholds leading to an empty set of objects falling into one of the subtrees are not considered.
        * As thresholds, we take the average of two values of the attribute (when sorting)
        * The behavior of the function in the case of a constant attribute can be any.
        * With the same Gini increments, I choose the minimum split.

        :param feature_vector: real-valued vector of attribute values
        :param target_vector: vector of object classes, len(feature_vector) == len(target_vector)

        :return thresholds: an ascending vector with all possible thresholds by which objects can be
         split into two different subsamples, or subtree
        :return ginis: vector with Gini criterion values for each of the thresholds in thresholds len(ginis) == len(thresholds)
        :return threshold_best: optimal threshold (number)
        :return gini_best: optimal value of the Gini criterion (number)
        """
    total_vector = np.column_stack((feature_vector, target_vector))
    total_vector = total_vector[total_vector[:, 0].argsort()]

    feature_values = np.unique(total_vector[:, 0])
    thresholds = (feature_values[:-1] + feature_values[1:]) / 2

    feature_vector_bound = np.append(total_vector[:, 0], total_vector[:, 0][-1])
    feature_vector_bound = np.array(feature_vector_bound)

    target = total_vector[:, 1]
    source_node = np.cumsum(np.ones(len(target)))
    left_node_1 = np.cumsum(target)
    left_node_0 = source_node - left_node_1
    left_node = left_node_1 + left_node_0

    right_node_1 = np.sum(target) - left_node_1
    right_node_0 = (len(target) - np.sum(target)) - left_node_0
    right_node = right_node_1 + right_node_0

    left_share = left_node / len(target)
    right_share = right_node / len(target)
    H_left = 1 - (left_node_1 / left_node) ** 2 - (left_node_0 / left_node) ** 2
    H_right = 1 - (right_node_1 / right_node) ** 2 - (right_node_0 / right_node) ** 2

    ginis = - left_share * H_left - right_share * H_right
    ginis = ginis[(feature_vector_bound[:-1] - feature_vector_bound[1:]) != 0]

    gini_best = np.max(ginis)
    threshold_best = thresholds[np.argmax(ginis)]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._achieved_depth = 0

    def _fit_node(self, sub_X, sub_y, node, depth):
        if (self._min_samples_split is not None) and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if (self._max_depth is not None) and (depth > self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # If all object in node has the same class
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if depth > self._achieved_depth:
            self._achieved_depth = depth

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        if self._min_samples_leaf is not None:
            if feature_best is None or len(sub_y[split]) < self._min_samples_leaf or len(
                    sub_y[np.logical_not(split)]) < self._min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth=(depth+1))
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth=(depth+1))

    def _predict_node(self, x, node):
        def next_node(node_cur):
            if 'threshold' in node_cur:
                if x[node_cur['feature_split']] < node_cur['threshold']:
                    return node_cur['left_child']
                return node_cur['right_child']
            elif x[node_cur['feature_split']] in node_cur['categories_split']:
                return node_cur['left_child']
            return node_cur['right_child']

        current_node = next_node(node)
        while current_node['type'] != 'terminal':
            current_node = next_node(current_node)

        return current_node['class']

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
