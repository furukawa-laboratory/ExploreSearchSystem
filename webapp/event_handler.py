import dash
from dash.dependencies import Input, Output, State
from webapp import app, logger
from webapp.figure_maker import (
    make_figure, prepare_materials, get_bmu,
    PAPER_COLOR, WORD_COLOR,
)

@app.callback(
    Output('paper-map', 'figure'),
    Output('word-map', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('viewer-selector', 'value'),
        Input('paper-map', 'clickData'),
        Input('word-map', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('paper-map', 'figure'),
        State('word-map', 'figure'),
])

def load_learning(n_clicks, viewer_name, p_clickData, w_clickData, keyword, p_prev_fig, w_prev_fig):
    logger.debug(f"p_clickData: {p_clickData}")
    logger.debug(f"w_clickData: {w_clickData}")
    ctx = dash.callback_context
    print(ctx.triggered[0]['prop_id'])
    print(type(ctx.triggered[0]['prop_id']))

    viewer_1_name, viewer_2_name = viewer_name, viewer_name
    if ctx.triggered[0]['prop_id'] == 'paper-map.clickData':
        if p_clickData and "points" in p_clickData and "pointIndex" in p_clickData["points"][0]:
            viewer_2_name = 'CCP'
    elif ctx.triggered[0]['prop_id'] == 'word-map.clickData':
        if w_clickData and "points" in w_clickData and "pointIndex" in w_clickData["points"][0]:
            viewer_1_name = 'CCP'

    keyword = keyword or "Machine Learning"
    return make_figure(keyword, viewer_1_name, 'viewer_1', w_clickData), make_figure(keyword, viewer_2_name, 'viewer_2', p_clickData)


@app.callback([
        Output('search-form', 'value'),
        Output('landing-search-form', 'value'),
    ], [
        Input('landing-explore-start', 'n_clicks'),
        Input('word-addition-popover-button', 'children'),
    ], [
        State('search-form', 'value'),
        State('landing-search-form', 'value'),
    ], prevent_initial_call=True)
def overwrite_search_form_value(n_clicks, popup_text, search_form, landing_form):
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

    return main_style, landing_style, paper_style, word_style, 'CCP'



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
        Output('word-addition-popover', 'is_open'),
        Output('word-addition-popover-button', 'children'),
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

    df, labels, _, history, _ = prepare_materials(keyword, 'TSOM')
    Z2 = history['Z2']
    paper_labels = labels[0].values.tolist()
    word_labels = labels[1].tolist()
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
        popup_text = f"{word} を検索キーワードに追加して検索！"
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

    return title, layout, style, should_popover_open, popup_text
