import dash
from dash.dependencies import Input, Output, State
from webapp import app, logger
from webapp.figure_maker import (
    make_figure, prepare_materials, get_bmu,
    PAPER_COLOR, WORD_COLOR, draw_toi,
)

@app.callback([
        Output('memory', 'data'),
        Output('paper-map-loading', 'loading-state'),
        Output('word-map-loading', 'loading-state'),
    ],
    [
        Input('explore-start', 'n_clicks'),
        Input('landing-explore-start', 'n_clicks'),
    ],
    [
        State('search-form', 'value'),
        State('landing-search-form', 'value'),
        State('memory', 'data'),
        State('published-date-limit', 'value')
])
def load_learning(n_clicks, n_clicks2, keyword, landing_keyword, data, within_5years):
    within_5years = True if within_5years == 'YES' else False
    logger.info('load_learning called')
    keyword = keyword or landing_keyword or "Machine Learning"
    df, labels, X, history, rank, umatrix_hisotry = prepare_materials(keyword, 'TSOM', within_5years)
    data = data or dict()
    data.update(
        snippet=df['snippet'].tolist(),
        url=df['URL'].tolist(),
        ranking=df['ranking'].tolist(),
        year=df['year'].tolist(),
        history=history,
        umatrix_hisotry=umatrix_hisotry,
        X=X,
        rank=rank,
        labels=labels,
    )
    return data, dict(is_loading=True), dict(is_loading=True)


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
    is_papermap_clicked = ctx.triggered[0]['prop_id'] == 'paper-map.clickData'
    is_wordmap_clicked  = ctx.triggered[0]['prop_id'] == 'word-map.clickData'
    if is_papermap_clicked:
        if p_clickData and "points" in p_clickData and "pointIndex" in p_clickData["points"][0]:
            viewer_2_name = 'CCP'
    elif is_wordmap_clicked:
        if w_clickData and "points" in w_clickData and "pointIndex" in w_clickData["points"][0]:
            viewer_1_name = 'CCP'

    history = data['history']
    umatrix_hisotry = data['umatrix_hisotry']
    X = data['X']
    rank = data['rank']
    labels = data['labels']
    logger.debug('learned data loaded.')
    history = {key: np.array(val) for key, val in history.items()}
    X = np.array(X)
    paper_fig = make_figure(history, umatrix_hisotry, X, rank, labels, viewer_1_name, 'viewer_1', w_clickData)
    word_fig  = make_figure(history, umatrix_hisotry, X, rank, labels, viewer_2_name, 'viewer_2', p_clickData)
    if viewer_2_name == 'CCP' and p_clickData:
        paper_fig = draw_toi(paper_fig, p_clickData, viewer_1_name)
    if viewer_1_name == 'CCP' and w_clickData:
        word_fig = draw_toi(word_fig, w_clickData, viewer_2_name)

    return paper_fig, word_fig


@app.callback([
        Output('search-form', 'value'),
        Output('landing-search-form', 'value'),
    ], [
        Input('landing-explore-start', 'n_clicks'),
        Input('word-addition-popover-button', 'n_clicks'),
    ], [
        State('word-addition-popover-button', 'children'),
        State('search-form', 'value'),
        State('landing-search-form', 'value'),
    ], prevent_initial_call=True)
def overwrite_search_form_value(n_clicks1, n_clicks2, popup_text, search_form, landing_form):
    if landing_form != '':  # first search
        search_form = landing_form or 'Machine Learning'
        logger.debug(f"search_form: {search_form}")
        return search_form, ''
    elif popup_text != '':  # additional search
        word = popup_text.split(' ')[0]
        return search_form + f' "{word}"', ''
    else:
        return search_form, landing_form


@app.callback([
        Output('main', 'style'),
        Output('landing', 'style'),
        Output('paper-map-col', 'style'),
        Output('word-map-col', 'style'),
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

    return main_style, landing_style, paper_style, word_style



import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
from scipy.spatial import distance as dist


def make_paper_component(title, abst, url, rank, year):
    return dbc.Card([
        dbc.CardBody([
        html.A(
            title,
            href=url,
            target='blank',
            className='display-6 text-dark',
            style=dict(fontSize='1.5rem')
        ),
        html.Span(
            rank,
            style=dict(
                verticalAlign='top',
            ),
        ),
        html.Span(
            f"（{year}年）",
        )]),
        dbc.CardFooter(abst)
    ], style=dict(
        marginBottom='10px',
        filter='drop-shadow(0px 8px 8px rgba(0, 0, 0, 0.25))',
    ))


@app.callback([
        Output('paper-list-title', 'children'),
        Output('paper-list-components', 'children'),
        Output('paper-list', 'style'),
        Output('word-addition-popover', 'is_open'),
        Output('word-addition-popover-button', 'children'),
    ],
    [
        Input('paper-map', 'clickData'),
        Input('word-map', 'clickData'),
    ],
    [
        State('paper-list', 'style'),
        State('memory', 'data'),
    ],
    prevent_initial_call=True
)
def make_paper_list(paperClickData, wordClickData, style, data):
    logger.debug('make_paper_list')

    ctx = dash.callback_context
    map_name = ctx.triggered[0]['prop_id'].split('.')[0]
    logger.info(f"map_name: {map_name}")

    history = data['history']
    logger.debug('learned data loaded.')
    history = {key: np.array(val) for key, val in history.items()}
    Z2 = history['Z2']

    paper_labels, word_labels = data['labels']
    if map_name == 'paper-map':
        should_popover_open = False
        clicked_point = [[paperClickData['points'][0]['x'], paperClickData['points'][0]['y']]] if paperClickData else [[0, 0]]
        clicked_point = np.array(clicked_point)
        dists = dist.cdist(history['Z1'], clicked_point)
        paper_idxs = np.argsort(dists, axis=0)[:3].flatten()
        title = "クリックした付近の論文"
        popup_text = ''
    else:
        should_popover_open = True
        clicked_point = [[wordClickData['points'][0]['x'], wordClickData['points'][0]['y']]] if wordClickData else [[0, 0]]
        clicked_point = np.array(clicked_point)
        logger.debug(clicked_point)
        bmu = get_bmu(history['Zeta'], wordClickData)
        y = history['Y'][:, bmu]
        word_idx = np.argmin(dist.cdist(Z2, history['Zeta'][bmu][None, :]), axis=0)
        logger.debug(f"word_idx: {word_idx}")
        word = word_labels[word_idx[0]]
        title = f"{word} を多く含む論文"
        popup_text = f"{word} を検索キーワードに追加！"
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
        make_paper_component(
            paper_labels[i],
            data['snippet'][i],
            data['url'][i],
            data['ranking'][i],
            data['year'][i]
        ) for i in paper_idxs
    ]
    style['backgroundColor'] = PAPER_COLOR if map_name == 'paper-map' else WORD_COLOR

    return title, layout, style, should_popover_open, popup_text
