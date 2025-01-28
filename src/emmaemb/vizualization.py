import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from src.emmaemb.core import Emma


def update_fig_layout(fig: go.Figure) -> go.Figure:
    """Update the layout of a plotly figure to adjust the font, line,\
        and grid settings.

    Args:
        fig (_type_): Plotly figure object.

    Returns:
        go.Figurge : Plotly figure object with updated layout.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="black"),
    )
    # show line at y=0 and x=0
    fig.update_xaxes(showline=True, linecolor="black", linewidth=2)
    fig.update_yaxes(showline=True, linecolor="black", linewidth=2)
    # hide gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_emb_space(
    emma: Emma,
    emb_space: str,
    method: str = "PCA",
    normalise: bool = True,
    color_by: str = None,
    random_state: int = 42,
    perplexity: int = 30,
) -> go.Figure:

    emma._check_for_emb_space(emb_space)
    embeddings = emma.emb[emb_space]["emb"]

    if color_by:
        if not emma._check_column_is_categorical(color_by):
            raise ValueError(f"Column {color_by} is not categorical")

    if normalise:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    if method == "PCA":
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        variance_explained = pca.explained_variance_ratio_
    elif method == "TSNE":
        tsne = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )
        embeddings_2d = tsne.fit_transform(embeddings)
    elif method == "UMAP":
        umap = UMAP(n_components=2, random_state=random_state)
        embeddings_2d = umap.fit_transform(embeddings)

    else:
        raise ValueError(f"Method {method} not implemented")

    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=emma.feature_data[color_by] if color_by else None,
        labels={"color": color_by},
        title=f"{emb_space} embeddings after {method}",
        hover_data={"Sample": emma.sample_names},
        opacity=0.5,
        color_discrete_map=(emma.color_map[color_by] if color_by else None),
    )

    fig.update_layout(
        width=800,
        height=800,
        autosize=False,
        legend=dict(
            title=f"{color_by.capitalize() if color_by else 'Sample'}",
        ),
    )

    fig.update_traces(
        marker=dict(size=max(10, (1 / len(emma.sample_names)) * 400))
    )

    if method == "PCA":
        fig.update_layout(
            xaxis_title="PC1 ({}%)".format(
                round(variance_explained[0] * 100, 2)
            ),
            yaxis_title="PC2 ({}%)".format(
                round(variance_explained[1] * 100, 2)
            ),
        )
    fig = update_fig_layout(fig)
    return fig
