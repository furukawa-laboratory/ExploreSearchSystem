import dash
from dash.dependencies import Input, Output, State
from webapp import app, logger
from webapp.figure_maker import (
    make_figure, prepare_materials, get_bmu,
    PAPER_COLOR, WORD_COLOR,
)

@app.callback([
        Output('memory', 'data'),
    ],
    [
        Input('explore-start', 'n_clicks'),
        Input('landing-explore-start', 'n_clicks'),
    ],
    [
        State('search-form', 'value'),
        State('memory', 'data'),
])
def load_learning(n_clicks, n_clicks2, keyword, data):
    keyword = keyword or "Machine Learning"
    _, labels, X, history, rank, umatrix_hisotry = prepare_materials(keyword, 'TSOM')
    data = data or dict()
    data.update(
        history=history,
        umatrix_hisotry=umatrix_hisotry,
        X=X,
        rank=rank,
        labels=labels,
    )
    return data,


@app.callback([
        Output('paper-map', 'figure'),
        Output('word-map', 'figure'),
    ],
    [
        Input('memory', 'modified_timestamp'),
        Input('viewer-selector', 'value'),
        Input('paper-map', 'clickData'),
        Input('word-map', 'clickData'),
    ],
    [
        State('memory', 'data'),
], prevent_initial_call=True)
def draw_maps(_, viewer_name, p_clickData, w_clickData, data):
    logger.debug(f"p_clickData: {p_clickData}")
    logger.debug(f"w_clickData: {w_clickData}")
    viewer_1_name, viewer_2_name = viewer_name, viewer_name
    ctx = dash.callback_context
    logger.debug(ctx.triggered[0]['prop_id'])
    logger.debug(type(ctx.triggered[0]['prop_id']))
    if ctx.triggered[0]['prop_id'] == 'paper-map.clickData':
        if p_clickData and "points" in p_clickData and "pointIndex" in p_clickData["points"][0]:
            viewer_2_name = 'CCP'
    elif ctx.triggered[0]['prop_id'] == 'word-map.clickData':
        if w_clickData and "points" in w_clickData and "pointIndex" in w_clickData["points"][0]:
            viewer_1_name = 'CCP'

    history, umatrix_hisotry, X, rank, labels = data['history'], data['umatrix_hisotry'], data['X'], data['rank'], data['labels']
    logger.debug('learned data loaded.')
    history = {key: np.array(val) for key, val in history.items()}
    X = np.array(X)
    paper_fig = make_figure(history, umatrix_hisotry, X, rank, labels, viewer_1_name, 'viewer_1', w_clickData)
    word_fig  = make_figure(history, umatrix_hisotry, X, rank, labels, viewer_2_name, 'viewer_2', p_clickData)
    return paper_fig, word_fig


@app.callback([
        Output('main', 'style'),
        Output('landing', 'style'),
        Output('paper-map-col', 'style'),
        Output('word-map-col', 'style'),
        Output('search-form', 'value'),
        Output('viewer-selector', 'value'),
    ], [
        Input('landing-explore-start', 'n_clicks'),
    ], [
        State('landing-search-form', 'value'),
    ], prevent_initial_call=True)
def make_page(n_clicks, keyword):
    logger.info(f"first search started with keyword: {keyword}")
    main_style = {}
    landing_style = {}
    paper_style = {"height": "100%"}
    word_style = {"height": "100%"}

    main_style['display'] = 'block'
    landing_style['display'] = 'none'
    paper_style['display'] = 'block'
    word_style['display'] = 'block'

    keyword = keyword or 'Machine Learning'

    return main_style, landing_style, paper_style, word_style, keyword, 'CCP'



import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
from scipy.spatial import distance as dist


def make_paper_component(title, abst, url, rank):
    return html.Div([
        rank,
        html.A(title, href=url, target='blank'),
        html.P(abst)
    ])


@app.callback([
        Output('paper-list-title', 'children'),
        Output('paper-list-components', 'children'),
        Output('paper-list-components', 'style'),
    ],
    [
        Input('paper-map', 'clickData'),
        Input('word-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('paper-list-components', 'style'),
    ],
    prevent_initial_call=True
)
def make_paper_list(paperClickData, wordClickData, keyword, style):
    logger.debug('make_paper_list')

    ctx = dash.callback_context
    map_name = ctx.triggered[0]['prop_id'].split('.')[0]
    logger.info(f"map_name: {map_name}")

    df, labels, _, history, _, _ = prepare_materials(keyword, 'TSOM')
    Z2 = history['Z2']
    paper_labels = labels[0].values.tolist()
    word_labels = labels[1].tolist()
    if map_name == 'paper-map':
        clicked_point = [[paperClickData['points'][0]['x'], paperClickData['points'][0]['y']]] if paperClickData else [[0, 0]]
        clicked_point = np.array(clicked_point)
        dists = dist.cdist(history['Z1'], clicked_point)
        paper_idxs = np.argsort(dists, axis=0)[:3].flatten()
        title = "クリックした付近の論文"
    else:
        clicked_point = [[wordClickData['points'][0]['x'], wordClickData['points'][0]['y']]] if wordClickData else [[0, 0]]
        clicked_point = np.array(clicked_point)
        logger.debug(clicked_point)
        bmu = get_bmu(history['Zeta'], wordClickData)
        y = history['Y'][:, bmu]
        word_idx = np.argmin(dist.cdist(Z2, history['Zeta'][bmu][None, :]), axis=0)
        logger.debug(f"word_idx: {word_idx}")
        word = word_labels[word_idx[0]]
        title = f"{word} を多く含む論文"
        target_nodes = (-y).flatten().argsort()[:3]
        logger.debug(f"target_nodes: {target_nodes}")
        paper_idxs = []
        for k in target_nodes:
            _dists = dist.cdist(history['Z1'], history['Zeta'][k, None])
            _paper_idxs = np.argsort(_dists, axis=0)[:3].flatten().tolist()
            paper_idxs.extend(_paper_idxs)
        seen = set()
        seen_add = seen.add
        paper_idxs = [idx for idx in paper_idxs if not (idx in seen or seen_add(idx))]
    logger.debug(f"Paper indexes {paper_idxs}")
    layout = [
        make_paper_component(paper_labels[i], df['snippet'][i], df['URL'][i], df['ranking'][i]) for i in paper_idxs
    ]
    style['borderColor'] = PAPER_COLOR if map_name == 'paper-map' else WORD_COLOR

    return title, layout, style
