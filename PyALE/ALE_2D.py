import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from lib import quantile_ied

def aleplot_2D_continuous(X, model, features, grid_size = 40):
  quantiles = np.append(0, np.arange(1/grid_size, 1+1/grid_size, 1/grid_size))
  bins_0 = [X[feature[0]].min()] + quantile_ied(X[feature[0]], quantiles).to_list()
  bins_0 = np.unique(bins_0)
  feat_cut_0 = pd.cut(X[feature[0]], bins_0, include_lowest=True)
  bin_codes_0 = feat_cut_0.cat.codes
  bin_codes_unique_0 = np.unique(bin_codes_0)
  
  bins_1 = [X[feature[1]].min()] + quantile_ied(X[feature[1]], quantiles).to_list()
  bins_1 = np.unique(bins_1)
  feat_cut_1 = pd.cut(X[feature[1]], bins_1, include_lowest=True)
  bin_codes_1 = feat_cut_1.cat.codes
  bin_codes_unique_1 = np.unique(bin_codes_1)
  
  X11 = X.copy()
  X12 = X.copy()
  X21 = X.copy()
  X22 = X.copy()
  
  X11[feature] = pd.DataFrame({feature[0]:[bins_0[i] for i in bin_codes_0],
                               feature[1]:[bins_1[i] for i in bin_codes_1]})
  X12[feature] = pd.DataFrame({feature[0]:[bins_0[i] for i in bin_codes_0],
                               feature[1]:[bins_1[i+1] for i in bin_codes_1]})
  X21[feature] = pd.DataFrame({feature[0]:[bins_0[i+1] for i in bin_codes_0],
                               feature[1]:[bins_1[i] for i in bin_codes_1]})
  X22[feature] = pd.DataFrame({feature[0]:[bins_0[i+1] for i in bin_codes_0],
                               feature[1]:[bins_1[i+1] for i in bin_codes_1]})
  
  y_11 = model.predict(X11)
  y_12 = model.predict(X12)
  y_21 = model.predict(X21)
  y_22 = model.predict(X22)
  
  delta_df = pd.DataFrame({feature[0]:bin_codes_0 + 1,
                          feature[1]:bin_codes_1 + 1,
                          'Delta':(y_22 - y_21) - (y_12 - y_11)})
  index_combinations = pd.MultiIndex.from_product([bin_codes_unique_0+1, bin_codes_unique_1+1], names=feature)
  
  delta_df = delta_df.groupby([feature[0], feature[1]]).Delta.agg(['size', 'mean'])
  
  sizes_df = delta_df['size'].reindex(index_combinations, fill_value=0)
  sizes_0 = sizes_df.groupby(level=0).sum().reindex(range(len(bins_0)), fill_value=0)
  sizes_1 = sizes_df.groupby(level=1).sum().reindex(range(len(bins_0)), fill_value=0)
  
  eff_df = delta_df['mean'].reindex(index_combinations, fill_value=np.nan)
  
  # ============== fill in the effects of missing combinations ================= #
  # ============== with kd-tree nearest neighbour algorithm   ================== #
  row_na_idx = np.where(eff_df.isna())[0]
  feat0_code_na = eff_df.iloc[row_na_idx].index.get_level_values(0)
  feat1_code_na = eff_df.iloc[row_na_idx].index.get_level_values(1)
  
  row_notna_idx = np.where(eff_df.notna())[0]
  feat0_code_notna = eff_df.iloc[row_notna_idx].index.get_level_values(0)
  feat1_code_notna = eff_df.iloc[row_notna_idx].index.get_level_values(1)
  
  if(len(row_na_idx) != 0):
    range0 = bins_0.max() - bins_0.min()
    range1 = bins_1.max() - bins_1.min()
    
    feats_at_na = pd.DataFrame({
      feature[0]:(bins_0[feat0_code_na - 1] + bins_0[feat0_code_na])/(2*range0),
      feature[1]:(bins_1[feat1_code_na - 1] + bins_1[feat1_code_na])/(2*range1)})
    feats_at_notna = pd.DataFrame({
      feature[0]:(bins_0[feat0_code_notna - 1] + bins_0[feat0_code_notna])/(2*range0),
      feature[1]:(bins_1[feat1_code_notna - 1] + bins_1[feat1_code_notna])/(2*range1)})
    # fit the algorithm with the features where the effect is not missing
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(feats_at_notna)
    # find the neighbours of the features where the effect is missing
    distances, indices = nbrs.kneighbors(feats_at_na)
    # fill the missing effects with the effects of the nearest neighbours
    eff_df.iloc[row_na_idx] =  eff_df.iloc[row_notna_idx[indices.flatten()]].to_list()
  
  # ============== cumulative sum of the difference ================= #
  eff_df = eff_df.groupby(level=0).cumsum().groupby(level=1).cumsum()
  
  # ============== centering with the moving average ================= #
  # subtract the cumulative sum of the 1D moving average (for each axis)
  eff_df_0 = eff_df - eff_df.groupby(level=1).shift(periods=1, axis=0, fill_value=0)
  fJ0 = (sizes_df * (eff_df_0.groupby(level=0).shift(periods=1, axis=0, fill_value=0) + eff_df_0)/2).sum(level=0).div(sizes_0).fillna(0).cumsum()
  
  eff_df_1 = eff_df - eff_df.groupby(level=0).shift(periods=1, axis=0, fill_value=0)
  fJ1 = (sizes_df * (eff_df_1.groupby(level=1).shift(periods=1, fill_value=0) + eff_df_1)/2).sum(level=1).div(sizes_1).fillna(0).cumsum()
  
  all_combinations = pd.MultiIndex.from_product([[x for x in range(len(bins_0))], [x for x in range(len(bins_1))]], names=feature)
  eff_df = eff_df.reindex(all_combinations, fill_value=0)
  eff_df = eff_df.subtract(fJ0, level=0).subtract(fJ1, level=1)
  
  # subtract the total average of a 2D moving average of size 2 (4 cells)
  idx = pd.IndexSlice
  eff_df = eff_df - \
  (sizes_df * \
  (eff_df.loc[idx[0:len(bin_codes_unique_0)-1], idx[0:len(bin_codes_unique_1)-1], :].values + \
  eff_df.loc[idx[1:len(bin_codes_unique_0)], idx[1:len(bin_codes_unique_1)], :].values + \
  eff_df.loc[idx[0:len(bin_codes_unique_0)-1], idx[1:len(bin_codes_unique_1)], :].values + \
  eff_df.loc[idx[1:len(bin_codes_unique_0)], idx[0:len(bin_codes_unique_1)-1], :].values)/4) \
  .sum()/sizes_df.sum()
  
  # renaming and preparing final output
  eff_df = eff_df.reset_index(name='eff')
  eff_df[features[0]] = bins_0[eff_df[features[0]]]
  eff_df[features[1]] = bins_1[eff_df[features[1]]]
  eff_grid = eff_df.pivot_table(columns=features[1], values='eff', index=features[0])
  
  return(eff_grid)
