import pandas as pd
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from matplotlib import pyplot as plt


def plot_with_data(df: pd.DataFrame, private: bool = False, percentage: float = 0.98):
    plt.cla()
    plt.clf()
    if private:
        df = df[df['private']]
    else:
        df = df[~df['private']]

    fig, axs = plt.subplots(1, 3, figsize=(40, 10))
    for idx, repo_type in enumerate(['space', 'model', 'dataset']):
        df_type = df[df['repo_type'] == repo_type]
        if len(df_type) == 0:
            axs[idx].set_title(f'No {repo_type.capitalize()} ({"Private" if private else "Public"}) Storages')
            axs[idx].axis('off')
        else:
            df_sorted = df_type.sort_values('total_size', ascending=False)
            df_sorted['cumulative_percentage'] = df_sorted['total_size'].cumsum() / df_sorted['total_size'].sum()
            df_percentage = df_sorted[df_sorted['cumulative_percentage'] <= percentage]
            others_sum = df_sorted['total_size'].sum() - df_percentage['total_size'].sum()
            plot_data = pd.concat([df_percentage, pd.DataFrame({'repo_id': ['Others'], 'total_size': [others_sum]})])
            axs[idx].pie(plot_data['total_size'], labels=plot_data['repo_id'], autopct='%1.1f%%', startangle=90)
            axs[idx].set_title(f'{repo_type.capitalize()} ({"Private" if private else "Public"}) Storages\n'
                               f'({plural_word(len(df_type), "repository")}, '
                               f'{size_to_bytes_str(int(df_type["total_size"].sum()), sigfigs=3, system="si")})')
            axs[idx].axis('equal')
