import numpy as np
import torch
from scipy import stats
import pandas as pd
from graphical_tools import Graphical_tools


class Correlation_tools:
    CIFAR10_FEATURES_DICT = {
        'train':
            {
                'simCLR': '../scan/results/cifar-10/pretext/features_seed1.npy',  ## reg simCLR
                'PCA': '../representations_results/results/PCA/simCLR_cifar10_scaled_k42/train1.npy',  ## after PCA
                'AE': '../representations_results/results/AE/cifar10/embedded_cifar10_lat512_train1.npy',
                'ViT': '../representations_results/results/ViT/cifar10/train1.npy',
                'GAN': '../representations_results/results/GAN/cifar10/train1.npy'
            },
        'test':
            {
                'simCLR': '../scan/results/cifar-10/pretext/test_features_seed1.npy',
                'PCA': '../representations_results/results/PCA/simCLR_cifar10_scaled_k42/test1.npy',
                'AE': '../representations_results/results/AE/cifar10/embedded_cifar10_lat512_test1.npy',
                'ViT': '../representations_results/results/ViT/cifar10/test1.npy',
                'GAN': '../representations_results/results/GAN/cifar10/test1.npy',
            }
    }

    @staticmethod
    def load_np(path):
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        res = np.load(path)
        np.load = np_load_old

        return list(res)

    @staticmethod
    def Spearman_rank_order_cor(lSet1, lSet2):
        """
        calculate Spearman rank-order correlation
        """

        def get_rank(lSet1, lSet2):
            """
            create two lists according to the order of lSet1. Add the missing images for each vector and add
            padding of the average score for the missing.
            """
            union = set(lSet1).union(set(lSet2))
            missing1 = union - set(lSet1)
            missing2 = union - set(lSet2)

            if len(missing1) != 0:
                missing1_pad = np.zeros(len(missing1)) + sum(range(len(lSet1) + 1, len(union) + 1)) / len(missing1)
                vector1 = list(range(1, len(lSet1) + 1)) + list(missing1_pad)
            else:
                vector1 = list(range(1, len(lSet1) + 1))

            if len(missing2) != 0:
                missing2_pad_value = sum(range(len(lSet2) + 1, len(union) + 1)) / len(missing2)
                vector2 = []

                for l in lSet1:
                    try:
                        val = lSet2.index(l) + 1
                    except ValueError:
                        val = missing2_pad_value
                    vector2.append(val)

                for l in missing1:
                    vector2.append(lSet2.index(l) + 1)

            else:
                vector2 = list(range(1, len(lSet2) + 1))

            return vector1, vector2

        print("original vec1, vec2 = ", lSet1, lSet2)
        vec1, vec2 = get_rank(lSet1, lSet2)
        print("ranked vec1, vec2 = ", vec1, vec2)
        return stats.pearsonr(vec1, vec2)

    @staticmethod
    def intersection(lSet1, lSet2):
        """
        finds the intersection between the sets
        """
        union = set(lSet1).union(set(lSet2))
        missing1 = union - set(lSet1)

        vec1 = []
        vec2 = []
        for l in lSet1:
            vec2.append(l in lSet2)
            vec1.append(l in lSet1)
        for l in missing1:
            vec2.append(l in lSet2)
            vec1.append(l in lSet1)

        return vec1, vec2, lSet1 + list(missing1)

    @staticmethod
    def binary_correlation(lSet1, lSet2):
        """
        finds the ratio between the amount of common items to the total items in the union
        """
        a, b, union = Correlation_tools.intersection(lSet1, lSet2)
        c = np.logical_and(a, b)
        return sum(c) / len(union)

    @staticmethod
    def correalation_matrix(vectors, cols):
        corr_list = np.zeros((len(vectors), len(vectors)))

        for i, s1 in enumerate(vectors):
            for j, s2 in enumerate(vectors):
                corr_list[i, j] = Correlation_tools.binary_correlation(s1, s2)

        df = pd.DataFrame(corr_list, columns=cols)
        return df

    @staticmethod
    def representations_diff(lSet1, lSet2, type1, type2):
        """
        gets two vectors of different types, change the sec to the first type and calculate distance
        """
        type1_rep = Correlation_tools.load_np(Correlation_tools.CIFAR10_FEATURES_DICT['train'][type1])
        type2_rep = Correlation_tools.load_np(Correlation_tools.CIFAR10_FEATURES_DICT['train'][type2])

        dif_matrix1 = np.zeros((len(lSet1), len(lSet2)))
        dif_matrix2 = np.zeros((len(lSet1), len(lSet2)))

        for i, l1 in enumerate(lSet1):
            l1_rep1 = type1_rep[l1]
            l1_rep2 = type2_rep[l1]

            for j, l2 in enumerate(lSet2):
                l2_rep1 = type1_rep[l2]
                l2_rep2 = type2_rep[l2]

                dif_matrix1[i, j] = np.linalg.norm(l2_rep1 - l1_rep1)
                dif_matrix2[i, j] = np.linalg.norm(l2_rep2 - l1_rep2)

        return pd.DataFrame(dif_matrix1), pd.DataFrame(dif_matrix2)

    @staticmethod
    def check_coverage(lSet2, uSet2, type1, delta1):
        """
        check the coverage of lSet2 using the space of lSet1
        """

        def construct_graph(batch_size=500):
            """
            -- from ProbCover code --
            creates a directed graph where:
            x->y iff l2(x,y) < delta.

            represented by a list of edges (a sparse matrix).
            stored in a dataframe
            """
            xs, ys, ds = [], [], []
            print(f'Start constructing graph using delta={delta1}')

            # distance computations are done in GPU
            cuda_feats = torch.tensor(rel_features).cuda()
            for i in range(len(all_features1) // batch_size):
                # distance comparisons are done in batches to reduce memory consumption
                cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
                dist = torch.cdist(cur_feats, cuda_feats)
                mask = dist < delta1

                # saving edges using indices list - saves memory.
                x, y = mask.nonzero().T
                xs.append(x.cpu() + batch_size * i)
                ys.append(y.cpu())
                ds.append(dist[mask].cpu())

            xs = torch.cat(xs).numpy()
            ys = torch.cat(ys).numpy()
            ds = torch.cat(ds).numpy()

            df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
            print(f'Finished constructing graph using delta={delta1}')
            print(f'Graph contains {len(df)} edges.')
            return df

        def calc_coverage(graph_df):
            # removing incoming edges to all covered samples from the existing labeled set
            edge_from_seen = np.isin(graph_df.x, np.arange(len(lSet2)))
            covered_samples = graph_df.y[edge_from_seen].unique()
            coverage = len(covered_samples)
            return coverage / len(rel_features)

        all_features1 = np.array(Correlation_tools.load_np(Correlation_tools.CIFAR10_FEATURES_DICT['train'][type1]))
        features = all_features1 / np.linalg.norm(all_features1, axis=1, keepdims=True)
        relevant_indices = np.concatenate([lSet2, uSet2]).astype(int)
        rel_features = features[relevant_indices]

        graph_df = construct_graph()
        return calc_coverage(graph_df)


########################################################################################################################
############# I used the following function to check several representations_results on active learning missions ###############
########################################################################################################################
def calc_coverage_on_diff_spaces():
    def cov_space():
        """
        get the coverage of each rep method on other methods
        """

        res = []
        for type1 in types:
            row_res = []
            for type2 in types:
                lSet1, uSet1 = Correlation_tools.load_np(types_sets['lSet'][type1]), Correlation_tools.load_np(
                    types_sets['uSet'][type1])
                lSet2, uSet2 = Correlation_tools.load_np(types_sets['lSet'][type2]), Correlation_tools.load_np(
                    types_sets['uSet'][type2])

                coverage = Correlation_tools.check_coverage(lSet2=lSet2, uSet2=uSet2, type1=type1, delta1=delta_dict[type1])
                row_res.append(coverage)
                # print(f'coverage of {type2} lSet on {type1} space: {coverage}')
            res.append(np.array(row_res))
        return res

    def show_heatmap(res):
        Graphical_tools.plot_heatmap(res, title="Coverage In Different Spaces", x_labels=list(types), y_labels=list(types))

    delta_dict = {'ViT': 0.4, 'PCA': 0.5, 'simCLR': 0.75, 'AE': 0.2, 'GAN': 0.8}
    types = delta_dict.keys()
    root_path = '../output/CIFAR10/resnet18/'
    types_sets: dict = {'lSet': {'ViT': f'{root_path}ViT_512_del0.4/episode_5/lSet.npy',
                                 'PCA': f'{root_path}PCA_delte0.5/episode_5/lSet.npy',
                                 'simCLR': f'{root_path}simCLR/episode_5/lSet.npy',
                                 'AE': f'{root_path}AEdelte0.2_512/episode_5/lSet.npy',
                                 'GAN': f'{root_path}GAN_del0.8/episode_5/lSet.npy'
                                 },
                        'uSet': {'ViT': f'{root_path}ViT_512_del0.4/episode_5/uSet.npy',
                                 'PCA': f'{root_path}PCA_delte0.5/episode_5/uSet.npy',
                                 'simCLR': f'{root_path}simCLR/episode_5/uSet.npy',
                                 'AE': f'{root_path}AEdelte0.2_512/episode_5/uSet.npy',
                                 'GAN': f'{root_path}GAN_del0.8/episode_5/uSet.npy'
                                 }
                        }

    result = np.array(cov_space())
    print(pd.DataFrame(result).to_markdown())
    show_heatmap(result)
########################################################################################################################
########################################################################################################################
########################################################################################################################
