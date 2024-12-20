import pandas as pd
from hbutils.system import urlsplit
from hfutils.operate import get_hf_client
from hfutils.repository import hf_hub_repo_analysis
from hfutils.utils import get_requests_session
from hfutils.utils.path import RepoTypeTyping
from huggingface_hub import configure_http_backend, get_session
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status
from tqdm import tqdm

configure_http_backend(get_requests_session)

from typing import Optional


def hf_hub_iter_lfs_files(repo_id: str, repo_type: str, endpoint: Optional[str] = None, hf_token: Optional[str] = None):
    session = get_session()
    cursor = None
    while True:
        r = session.get(
            f"{endpoint or ENDPOINT}/api/{repo_type}s/{repo_id}/lfs-files",
            headers=build_hf_headers(token=hf_token),
            params={} if not cursor else {'cursor': cursor}
        )
        hf_raise_for_status(r)
        yield from r.json()

        if r.links.get('next'):
            cursor = urlsplit(r.links['next']['url']).query_dict['cursor']
        else:
            break


def analysis_repo(repo_id: str, repo_type: RepoTypeTyping = 'dataset'):
    repo_info = hf_hub_repo_analysis(
        repo_id=repo_id,
        repo_type=repo_type,
    )
    lfs_count, lfs_size = 0, 0
    for item in hf_hub_iter_lfs_files(
            repo_id=repo_id,
            repo_type=repo_type,
    ):
        lfs_count += 1
        lfs_size += item['size']

    return {
        'repo_id': repo_id,
        'repo_type': repo_type,
        'files': len(repo_info),
        'total_size': repo_info.total_size,
        'lfs_files': lfs_count,
        'lfs_size': lfs_size,
    }


def hf_hub_scan_for_author(author: str, analysis_private: bool = True) -> pd.DataFrame:
    hf_client = get_hf_client()
    repos = []
    for repo_item in tqdm(list(hf_client.list_spaces(author=author)), desc=f'Scanning Spaces of {author!r}'):
        if repo_item.private and not analysis_private:
            continue
        repo_info = analysis_repo(
            repo_id=repo_item.id,
            repo_type='space',
        )
        repos.append({
            'repo_id': repo_item.id,
            'repo_type': 'space',
            'private': repo_item.private,
            **repo_info,
        })

    for repo_item in tqdm(list(hf_client.list_models(author=author)), desc=f'Scanning Models of {author!r}'):
        if repo_item.private and not analysis_private:
            continue
        repo_info = analysis_repo(
            repo_id=repo_item.id,
            repo_type='model',
        )
        repos.append({
            'repo_id': repo_item.id,
            'repo_type': 'model',
            'private': repo_item.private,
            **repo_info,
        })

    for repo_item in tqdm(list(hf_client.list_datasets(author=author)), desc=f'Scanning Datasets of {author!r}'):
        if repo_item.private and not analysis_private:
            continue
        repo_info = analysis_repo(
            repo_id=repo_item.id,
            repo_type='dataset',
        )
        repos.append({
            'repo_id': repo_item.id,
            'repo_type': 'dataset',
            'private': repo_item.private,
            **repo_info,
        })

    df_repos = pd.DataFrame(repos)
    df_repos = df_repos.sort_values(by=['lfs_size', 'lfs_files', 'total_size', 'files', 'repo_id', 'repo_type'],
                                    ascending=[False, False, False, False, True, True])

    return df_repos
