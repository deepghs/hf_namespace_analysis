import pandas as pd
from hfutils.operate import get_hf_client
from hfutils.repository import hf_hub_repo_analysis
from hfutils.utils import get_requests_session
from huggingface_hub import configure_http_backend
from tqdm import tqdm

configure_http_backend(get_requests_session)


def hf_hub_scan_for_author(author: str, analysis_private: bool = True) -> pd.DataFrame:
    hf_client = get_hf_client()
    repos = []
    for repo_item in tqdm(list(hf_client.list_spaces(author=author)), desc=f'Scanning Spaces of {author!r}'):
        if repo_item.private and not analysis_private:
            continue
        repo = hf_hub_repo_analysis(
            repo_id=repo_item.id,
            repo_type='space',
        )
        repos.append({
            'repo_id': repo_item.id,
            'repo_type': 'space',
            'private': repo_item.private,
            'files': len(repo),
            'total_size': repo.total_size,
        })

    for repo_item in tqdm(list(hf_client.list_models(author=author)), desc=f'Scanning Models of {author!r}'):
        if repo_item.private and not analysis_private:
            continue
        repo = hf_hub_repo_analysis(
            repo_id=repo_item.id,
            repo_type='model',
        )
        repos.append({
            'repo_id': repo_item.id,
            'repo_type': 'model',
            'private': repo_item.private,
            'files': len(repo),
            'total_size': repo.total_size,
        })

    for repo_item in tqdm(list(hf_client.list_datasets(author=author)), desc=f'Scanning Datasets of {author!r}'):
        if repo_item.private and not analysis_private:
            continue
        repo = hf_hub_repo_analysis(
            repo_id=repo_item.id,
            repo_type='dataset',
        )
        repos.append({
            'repo_id': repo_item.id,
            'repo_type': 'dataset',
            'private': repo_item.private,
            'files': len(repo),
            'total_size': repo.total_size,
        })

    df_repos = pd.DataFrame(repos)
    df_repos = df_repos.sort_values(by=['total_size', 'files', 'repo_id', 'repo_type'],
                                    ascending=[False, False, True, True])

    return df_repos
