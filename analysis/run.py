import logging
import os.path
from typing import Optional

import click
import matplotlib.pyplot as plt
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import get_requests_session, number_to_tag
from huggingface_hub import configure_http_backend, hf_hub_url

from .plot import plot_with_data
from .scan import hf_hub_scan_for_author

CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


@click.command(help='Run the analysis of Huggingface Namespace.',
               context_settings=CONTEXT_SETTINGS)
@click.option('-a', '--author', 'author', type=str, required=True,
              help='Author name (both user and organization are supported).', show_default=True)
@click.option('-r', '--repository', 'repository', type=str, default=None,
              help='Repository to save the result, default is {author}/storage_analysis.', show_default=True)
@click.option('--private', 'analysis_private', is_flag=True, type=bool, default=False,
              help='Analysis private repositories as well if possible', show_default=True)
def run(author: str, repository: Optional[str] = None, analysis_private: bool = False):
    configure_http_backend(get_requests_session)
    logging.basicConfig(level=logging.INFO)

    hf_client = get_hf_client()
    repository = repository or f'{author}/storage_analysis'
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset')

    df = hf_hub_scan_for_author(author, analysis_private=analysis_private)
    logging.info(f'Analysis Result:\n{df}')

    with TemporaryDirectory() as td:
        df.to_csv(os.path.join(td, 'repositories.csv'), index=False)
        df.to_parquet(os.path.join(td, 'repositories.parquet'), index=False)

        with open(os.path.join(td, 'README.md'), 'w') as f:
            print('---', file=f)
            print('license: other', file=f)
            print('tags:', file=f)
            print('- table', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df))}', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'# Storage Analysis of {author!r}', file=f)
            print(f'', file=f)

            print(f'## Public Repositories', file=f)
            print(f'', file=f)
            if len(df[~df['private']]) == 0:
                print('No public repository found.', file=f)
                print(f'', file=f)
            else:
                public_plot_file = os.path.join(td, 'plot_public.png')
                plot_with_data(df, private=False)
                plt.savefig(public_plot_file)
                print(
                    f'![public repos]({hf_hub_url(repo_id=repository, repo_type="dataset", filename=os.path.relpath(public_plot_file, td))})',
                    file=f)
                print(f'', file=f)

                for idx, repo_type in enumerate(['space', 'model', 'dataset']):
                    df_type = df[~df['private'] & (df['repo_type'] == repo_type)]
                    if len(df_type) > 0:
                        df_type_shown = df_type[:50]
                        df_type_shown['size'] = df_type_shown['total_size'].map(
                            lambda x: size_to_bytes_str(x, sigfigs=3, system="si"))
                        df_type_shown['link'] = [
                            f'[Link]({hf_hub_repo_url(repo_id=repo_id, repo_type=repo_type_)})'
                            for repo_id, repo_type_ in zip(df_type_shown['repo_id'], df_type_shown['repo_type'])
                        ]
                        print(f'### {repo_type.capitalize()} (Public)', file=f)
                        print(f'', file=f)
                        print(f'{plural_word(len(df_type), f"public {repo_type} repository")} in total, '
                              f'only the biggest {len(df_type_shown)} of them are listed here.', file=f)
                        print(f'', file=f)
                        print(f'Total storage cost of these repositories: '
                              f'{size_to_bytes_str(int(df_type["total_size"].sum()), sigfigs=3, system="si")}', file=f)
                        print(f'', file=f)
                        print(df_type_shown.to_markdown(index=False), file=f)
                        print(f'', file=f)

            if analysis_private:
                print(f'## Private Repositories', file=f)
                print(f'', file=f)
                if len(df[df['private']]) == 0:
                    print('No private repository found.', file=f)
                    print(f'', file=f)
                else:
                    private_plot_file = os.path.join(td, 'plot_private.png')
                    plot_with_data(df, private=True)
                    plt.savefig(private_plot_file)
                    print(
                        f'![private repos]({hf_hub_url(repo_id=repository, repo_type="dataset", filename=os.path.relpath(private_plot_file, td))})',
                        file=f)
                    print(f'', file=f)

                    for idx, repo_type in enumerate(['space', 'model', 'dataset']):
                        df_type = df[df['private'] & (df['repo_type'] == repo_type)]
                        if len(df_type) > 0:
                            df_type_shown = df_type[:50]
                            df_type_shown['size'] = df_type_shown['total_size'].map(
                                lambda x: size_to_bytes_str(x, sigfigs=3, system="si"))
                            df_type_shown['link'] = [
                                f'[Link]({hf_hub_repo_url(repo_id=repo_id, repo_type=repo_type_)})'
                                for repo_id, repo_type_ in zip(df_type_shown['repo_id'], df_type_shown['repo_type'])
                            ]
                            print(df_type_shown)
                            print(f'### {repo_type.capitalize()} (Private)', file=f)
                            print(f'', file=f)
                            print(f'{plural_word(len(df_type), f"private {repo_type} repository")} in total, '
                                  f'only the biggest {len(df_type_shown)} of them are listed here.', file=f)
                            print(f'', file=f)
                            print(f'Total storage cost of these repositories: '
                                  f'{size_to_bytes_str(int(df_type["total_size"].sum()), sigfigs=3, system="si")}',
                                  file=f)
                            print(f'', file=f)
                            print(df_type_shown.to_markdown(index=False), file=f)
                            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            path_in_repo='.',
            local_directory=td,
            message=f'Upload storage analysis of {author!r}',
        )


if __name__ == '__main__':
    run()
