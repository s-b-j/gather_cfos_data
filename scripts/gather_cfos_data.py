import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from matplotlib_venn import venn3
pd.options.mode.chained_assignment = None

working_dir = r"C:\Users\shane\workspace\gather_cfos_data\scripts"

exclusion_list = ["SJ0613", "SJ0608"]


def get_stack_paths(main_dir):
    # if stack_dir_save_path does not exist or needs updating, run the following:
    stack_dirs = []
    for path in os.listdir(main_dir):
        for subdir in os.listdir(main_dir + path):
            if (subdir.find("SJ") > 0) & (subdir.find("destriped") > 0):
                stack_dirs.append(main_dir + path + "/" + subdir + "/")
            else:
                pass
    # save stack_dirs
    # else:load stack dirs file
    return stack_dirs


def get_cfos_paths(stack_dirs, out_path):
    if os.path.exists(out_path):
        cfos_paths = pd.read_csv(out_path)
    else:
        cfos_paths = pd.DataFrame()
        for dir in stack_dirs:
            if ("Ex_647_Em_680_stitched" in os.listdir(dir)):
                cfos_dir = dir+"Ex_647_Em_680_stitched"
                file_list = os.listdir(cfos_dir)
                if file_list:
                    point_file = [i for i in file_list if "Points_in_region" in i]
                    if (point_file):
                        cfos_path = pd.DataFrame({"directory": os.path.join(cfos_dir, str(point_file[0]))}, index=[0])
                        cfos_paths = pd.concat([cfos_paths, cfos_path], axis=0)
        print("Saving cfos paths to file")
        cfos_paths.to_csv(out_path)
    return cfos_paths


def gather_cfos_files_horizontally(cfos_paths, groups, collapse_groups=True):
    print("Gathering cfos files and stacking horizontally")
    cfos_all = pd.DataFrame()
    cfos_paths["animal_num"] = ["SJ"+file.split("SJ")[1][:4] for file in cfos_paths.directory]
    cfos_paths_included = cfos_paths[np.logical_not(cfos_paths.animal_num.isin(exclusion_list))]
    if collapse_groups:
        for i, file in enumerate(cfos_paths_included.directory):
            animal_num = "SJ"+file.split("SJ")[1][:4]
            print(f"Gathering data from animal: {animal_num}")
            group_dict = groups.set_index("animal_num")["group_collapse"].to_dict()
            group = group_dict[animal_num]
            animal_group = group + "_" + animal_num
            if i == 0:
                cfos = pd.read_csv(file, usecols = ['name', 'acronym', 'density (cells/mm^3)'])
                cfos = cfos.rename(columns={"density (cells/mm^3)":animal_group})
                cfos_all = pd.concat([cfos_all, cfos], axis=1)
            else:
                cfos = pd.read_csv(file, usecols = ['density (cells/mm^3)'])
                cfos = cfos.rename(columns={"density (cells/mm^3)":animal_group})
                cfos_all = pd.concat([cfos_all, cfos], axis=1)
    else:
        for i, file in enumerate(cfos_paths_included.directory):
            animal_num = "SJ"+file.split("SJ")[1][:4]
            print(f"Gathering data from animal: {animal_num}")
            group_dict = groups.set_index("animal_num")["group"].to_dict()
            group = group_dict[animal_num]
            animal_group = group + "_" + animal_num
            if i == 0:
                cfos = pd.read_csv(file, usecols = ['name', 'acronym', 'density (cells/mm^3)'])
                cfos = cfos.rename(columns={"density (cells/mm^3)":animal_group})
                cfos_all = pd.concat([cfos_all, cfos], axis=1)
            else:
                cfos = pd.read_csv(file, usecols = ['density (cells/mm^3)'])
                cfos = cfos.rename(columns={"density (cells/mm^3)":animal_group})
                cfos_all = pd.concat([cfos_all, cfos], axis=1)
    return cfos_all


def gather_cfos_files_vertically(cfos_paths, groups, collapse_groups=True):
    print("Gathering cfos files and stacking vertically")
    cfos_all = pd.DataFrame()
    cfos_paths["animal_num"] = ["SJ"+file.split("SJ")[1][:4] for file in cfos_paths.directory]
    cfos_paths_included = cfos_paths[np.logical_not(cfos_paths.animal_num.isin(exclusion_list))]
    if collapse_groups:
        for i, file in enumerate(cfos_paths_included.directory):
            animal_num = "SJ"+file.split("SJ")[1][:4]
            print(f"Gathering data from animal: {animal_num}")
            group_dict = groups.set_index("animal_num")["group_collapse"].to_dict() # TODO: stop mapping the control group name to the group name
            group = group_dict[animal_num]
            group_animal = group + "_" + animal_num
            cfos = pd.read_csv(file)
            cfos["group"] = group
            cfos["group_animal"] = group_animal
            cfos_all = pd.concat([cfos_all, cfos], axis=0)
    else:
        for i, file in enumerate(cfos_paths_included.directory):
            animal_num = "SJ" + file.split("SJ")[1][:4]
            print(f"Gathering data from animal: {animal_num}")
            group_dict = groups.set_index("animal_num")["group"].to_dict()
            group = group_dict[animal_num]
            group_animal = group + "_" + animal_num
            cfos = pd.read_csv(file)
            cfos["group"] = group
            cfos["group_animal"] = group_animal
            cfos_all = pd.concat([cfos_all, cfos], axis=0)
    return cfos_all


def zscore(cfos_vrt, control_dict):
    row_z = []
    cfos_vrt_control_bool = ["FP" in i for i in cfos_vrt.group]
    cfos_vrt_control = cfos_vrt[cfos_vrt_control_bool]
    cfos_vrt_control.loc[:, "control_grp"] = cfos_vrt_control.group.map(control_dict).to_list()
    cfos_vrt_control_mean = cfos_vrt_control.groupby(["name", "control_grp"]).agg(
        density_mean=("density (cells/mm^3)", "mean"),
        density_std=("density (cells/mm^3)", "std"))
    cfos_vrt.loc[:, "control_grp"] = cfos_vrt.group.map(control_dict)
    for row in cfos_vrt.iterrows():
        ctrl = cfos_vrt_control_mean.loc[row[1]["name"], row[1].control_grp]
        row_z.append((row[1]["density (cells/mm^3)"] - ctrl.density_mean)/ctrl.density_std)
    return row_z


def boxplot(cfos_vrt_collapse, labels, palette):
    order = ["iTBS_30sn_YFP", "iTBS_30sn_ChR", "1sn_GFP", "iTBS_1sn_ChR", "cTBS_1sn_ChR"]
    if not isinstance(labels, list):
        labels = [labels]
    cfos_vrt_collapse_sub = cfos_vrt_collapse[cfos_vrt_collapse["name"].isin(labels)]
    fig, ax = plt.subplots(figsize=(10,6))
    # sns.set_style("white")
    plt.style.use('ggplot')
    sns.boxplot(data=cfos_vrt_collapse_sub, ax=ax, x="group", y="density_zscore", order=order, palette=palette)
    sns.swarmplot(data=cfos_vrt_collapse_sub, ax=ax, x="group", y="density_zscore", order=order, edgecolor="gray", linewidth=1, palette=palette)
    plt.xticks(rotation=45)
    plt.ylabel("Density")
    plt.subplots_adjust(bottom=0.2)
    ax.axhline(0, ls='--')
    plt.title(labels)
    plt.show()


def boxplot_multi(cfos_vrt_collapse, y_data, plot_label, labels, palette, save_path=r"C:\Users\shane\workspace\gather_cfos_data\results"):
    order = ["iTBS_30sn_YFP", "iTBS_30sn_ChR", "1sn_GFP", "iTBS_1sn_ChR", "cTBS_1sn_ChR"]
    if not isinstance(labels, list):
        labels = [labels]
    cfos_vrt_collapse_sub = cfos_vrt_collapse[cfos_vrt_collapse["name"].isin(labels)]
    # sns.set_style("white")
    g = sns.FacetGrid(cfos_vrt_collapse_sub, col="name", col_wrap=5, legend_out=True, height=4.5, aspect=1)
    plt.style.use('ggplot')
    g.map(sns.boxplot, "group", y_data, order=order, palette=palette)
    g.map(sns.swarmplot, "group", y_data, order=order, edgecolor="gray", linewidth=1, palette=palette)
    g.set_titles(col_template="{col_name}")
    for ax in g.axes:
        ax.axhline(0, ls='--')
    for axes in g.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
    g.fig.subplots_adjust(top=0.9, bottom=0.2)
    title = y_data + "_" + plot_label
    save_path_full = save_path + r"/" + title + ".png"
    save_path_full = save_path_full.replace(" ", "_").replace("^","")
    g.fig.suptitle(title, fontsize=16)
    print(save_path_full)
    g.savefig(save_path_full)
    return g


def run_ttests(cfos_vrt_collapse):
    region_list = cfos_vrt_collapse["name"].unique()
    t_test_df = pd.DataFrame({"region":region_list, "t_iTBS_30sn":"", "p_iTBS_30sn": "", "t_iTBS_1sn": "", "p_iTBS_1sn": "", "t_cTBS_1sn": "", "p_cTBS_1sn": ""})
    t_test_df = t_test_df.set_index("region")

    for i, region in enumerate(region_list):
        subset = cfos_vrt_collapse[cfos_vrt_collapse["name"] == region]
        iTBS_t_test_30sn = ttest_ind(subset[subset["group"] == "iTBS_30sn_ChR"].density_zscore, subset[subset["group"] == "iTBS_30sn_YFP"].density_zscore) 
        t_test_df.loc[region].t_iTBS_30sn = iTBS_t_test_30sn[0]
        t_test_df.loc[region].p_iTBS_30sn = iTBS_t_test_30sn[1]

        iTBS_t_test_1sn = ttest_ind(subset[subset["group"] == "iTBS_1sn_ChR"].density_zscore, subset[subset["group"] == "1sn_GFP"].density_zscore)
        t_test_df.loc[region].t_iTBS_1sn = iTBS_t_test_1sn[0]
        t_test_df.loc[region].p_iTBS_1sn = iTBS_t_test_1sn[1]

        cTBS_t_test_1sn = ttest_ind(subset[subset["group"] == "cTBS_1sn_ChR"].density_zscore, subset[subset["group"] == "1sn_GFP"].density_zscore)
        t_test_df.loc[region].t_cTBS_1sn = cTBS_t_test_1sn[0]
        t_test_df.loc[region].p_cTBS_1sn = cTBS_t_test_1sn[1]
    return t_test_df


def generate_venn_diagram(t_test_df, alpha=0.05):
    t_test_df_sort_iTBS_30sn = t_test_df.sort_values(by=["p_iTBS_30sn"])
    t_test_df_sort_iTBS_1sn = t_test_df.sort_values(by=["p_iTBS_1sn"])
    t_test_df_sort_cTBS_1sn = t_test_df.sort_values(by=["p_cTBS_1sn"])
    signif_iTBS_30sn = t_test_df[t_test_df["p_iTBS_30sn"] < alpha].reset_index().region.to_list()
    signif_cTBS_1sn = t_test_df[t_test_df["p_cTBS_1sn"] < alpha].reset_index().region.to_list()
    signif_iTBS_1sn = t_test_df[t_test_df["p_iTBS_1sn"] < alpha].reset_index().region.to_list()
    set_iTBS_30sn = set(signif_iTBS_30sn)
    set_iTBS_1sn = set(signif_iTBS_1sn)
    set_cTBS_1sn = set(signif_cTBS_1sn)
    venn_diagram = venn3(
        subsets = (
            len(set_iTBS_30sn.difference(set_iTBS_1sn.union(set_cTBS_1sn))),
            len(set_iTBS_1sn.difference(set_iTBS_30sn.union(set_cTBS_1sn))),
            len(set_iTBS_30sn.intersection(set_iTBS_1sn).difference(set_cTBS_1sn)),
            len(set_cTBS_1sn.difference(set_iTBS_1sn.union(set_iTBS_30sn))),
            len(set_iTBS_30sn.intersection(set_cTBS_1sn).difference(set_iTBS_1sn)),
            len(set_iTBS_1sn.intersection(set_cTBS_1sn).difference(set_iTBS_30sn)),
            len(set_iTBS_1sn.intersection(set_cTBS_1sn).intersection(set_iTBS_30sn))),
            set_labels=('iTBS_30session', 'iTBS_1session', 'cTBS_1session'),
            alpha = 0.5,
            )
    return venn_diagram, t_test_df_sort_iTBS_30sn, t_test_df_sort_iTBS_1sn, t_test_df_sort_cTBS_1sn


def main():
    optoTMS_colors = {
        "iTBS_30sn_ChR": '#FF8C00',
        "iTBS_1sn_ChR": '#FFE4C4',
        "iTBS_30sn_YFP": '#808080',
        "1sn_GFP": '#D3D3D3',
        "cTBS_1sn_ChR": '#6495ED',
        }

    control_dict = {
        "cTBS_1sn_ChR": "cTBS_1sn_GFP",
        "iTBS_1sn_ChR": "iTBS_1sn_GFP",
        "iTBS_30sn_ChR": "iTBS_30sn_YFP",
        "cTBS_1sn_GFP": "cTBS_1sn_GFP",
        "iTBS_1sn_GFP": "iTBS_1sn_GFP",
        "iTBS_30sn_YFP": "iTBS_30sn_YFP",
        "1sn_GFP": "1sn_GFP",
        }

    control_dict_collapse = {
        "cTBS_1sn_ChR": "1sn_GFP",
        "iTBS_1sn_ChR": "1sn_GFP",
        "iTBS_30sn_ChR": "iTBS_30sn_YFP",
        "cTBS_1sn_GFP": "1sn_GFP",
        "iTBS_1sn_GFP": "1sn_GFP",
        "1sn_GFP": "1sn_GFP",
        "iTBS_30sn_YFP":"iTBS_30sn_YFP",
        }
    
    prime_targets = [
        "left Prelimbic area",
        "right Prelimbic area",
        "left Caudoputamen",
        "right Caudoputamen",
        "left Nucleus accumbens",
        "right Nucleus accumbens",
        "left Infralimbic area",
        "right Infralimbic area",
        "left Mediodorsal nucleus of thalamus",
        "right Mediodorsal nucleus of thalamus",
    ]

    main_dir = r"Z:/SmartSPIM_Data/"
    out_path = r"C:\Users\shane\workspace\gather_cfos_data\data\cfos_dirs.csv"
    group_path = r"C:\Users\shane\ListonLab Dropbox\Shane Johnson\shane\python_projects\gather_cfos_data\docs\optotms_animal_num_group.csv"
    groups = pd.read_csv(group_path)
    stack_dirs = get_stack_paths(main_dir)
    cfos_paths = get_cfos_paths(stack_dirs, out_path)
    # cfos_hrz = gather_cfos_files_horizontally(cfos_paths, groups)
    cfos_vrt = gather_cfos_files_vertically(cfos_paths, groups, collapse_groups=False)
    cfos_vrt_collapse = gather_cfos_files_vertically(cfos_paths, groups, collapse_groups=True)
    # cfos_vrt["density_zscore"] = zscore(cfos_vrt, control_dict=control_dict)
    cfos_vrt_collapse["density_zscore"] = zscore(cfos_vrt, control_dict=control_dict_collapse)
    t_test_df = run_ttests(cfos_vrt_collapse)
    # generate_venn_diagram(t_test_df, alpha=0.05)
    # sns.boxplot(data=cfos_vrt[cfos_vrt["name"]=="left Prelimbic area"], x="group", y="density_zscore")
    (
        venn_diagram,
        sort_iTBS_30sn,
        sort_iTBS_1sn,
        sort_cTBS_1sn,
        ) = generate_venn_diagram(t_test_df, alpha=0.05)
    top_01_10_iTBS_30sn = sort_iTBS_30sn.iloc[0:10].index.to_list()
    top_11_20_iTBS_30sn = sort_iTBS_30sn.iloc[11:20].index.to_list()
    top_01_10_iTBS_1sn = sort_iTBS_1sn.iloc[0:10].index.to_list()
    top_11_20_iTBS_1sn = sort_iTBS_1sn.iloc[11:20].index.to_list()
    top_01_10_cTBS_1sn = sort_cTBS_1sn.iloc[0:10].index.to_list()
    top_11_20_cTBS_1sn = sort_cTBS_1sn.iloc[11:20].index.to_list()
    cfos_vrt_collapse = cfos_vrt_collapse.rename(columns={"density (cells/mm^3)":"density_cells_per_mm3"})
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_cells_per_mm3",
        plot_label="PL primary targets",
        labels=prime_targets,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_zscore",
        plot_label="PL major targets",
        labels=prime_targets,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_zscore",
        plot_label="Top hits (1-10) by iTBS 30 session p-value",
        labels=top_01_10_iTBS_30sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_cells_per_mm3",
        plot_label="Top hits (1-10) by iTBS 30 session p-value",
        labels=top_01_10_iTBS_30sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_zscore",
        plot_label="Top hits (11-20) by iTBS 30 session p-value",
        labels=top_11_20_iTBS_30sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_cells_per_mm3",
        plot_label="Top hits (11-20) by iTBS 30 session p-value",
        labels=top_11_20_iTBS_30sn,
        palette=optoTMS_colors,
        )

    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_zscore",
        plot_label="Top hits (1-10) by iTBS 1 session p-value",
        labels=top_01_10_iTBS_1sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_cells_per_mm3",
        plot_label="Top hits (1-10) by iTBS 1 session p-value",
        labels=top_01_10_iTBS_1sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_zscore",
        plot_label="Top hits (11-20) by iTBS 1 session p-value",
        labels=top_11_20_iTBS_1sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_cells_per_mm3",
        plot_label="Top hits (11-20) by iTBS 1 session p-value",
        labels=top_11_20_iTBS_1sn,
        palette=optoTMS_colors,
        )

    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_zscore",
        plot_label="Top hits (1-10) by cTBS 1 session p-value",
        labels=top_01_10_cTBS_1sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_cells_per_mm3",
        plot_label="Top hits (1-10) by cTBS 1 session p-value",
        labels=top_01_10_cTBS_1sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_zscore",
        plot_label="Top hits (11-20) by cTBS 1 session p-value",
        labels=top_11_20_cTBS_1sn,
        palette=optoTMS_colors,
        )
    boxplot_multi(
        cfos_vrt_collapse,
        y_data="density_cells_per_mm3",
        plot_label="Top hits (11-20) by cTBS 1 session p-value",
        labels=top_11_20_cTBS_1sn,
        palette=optoTMS_colors,
        )
    cfos_vrt_collapse_pivot = pd.pivot_table(cfos_vrt_collapse, values = "density_zscore", index="name", columns="group", aggfunc=np.nanmean)
    
if __name__ == "__main__":
    main()