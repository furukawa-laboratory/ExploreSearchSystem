# -*- coding: utf-8 -*-

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from webapp import app
from webapp.figure_maker import make_figure


# U-Matrix の説明用のモーダル
umatrix_modal = dbc.Modal([
    dbc.ModalHeader("U-Matrix 表示とは？"),
    dbc.ModalBody("青い領域がクラスタ，赤い領域がクラスタ境界を表す"),
    dbc.ModalFooter(
        dbc.Button(
            "Close", id="close-umatrix-modal", className="ml-auto", n_clicks=0
        )
    ),
], id="umatrix-modal", is_open=False, centered=True)


# app.callback(
#     Output('umatrix-modal', 'is_open'),
#     [
#         Input('open-umatrix-modal', 'n_clicks'),
#         Input('close-umatrix-modal', 'n_clicks'),
#     ],
#     State('umatrix-modal', 'is_open'))(toggle_modal)


link_card = dbc.Card([
    # html.P("", id="card-text", className="h4"),
    dbc.CardHeader("", id="card-text", className="h4"),
    html.P("", id="snippet-text", className="h5",style={"min-height":"100px"}),
    html.A(
        id='link',
        href='#',
        children="マウスを当ててみよう",
        target="_self",
        className="btn btn-outline-primary btn-lg",
    ),
    dbc.CardFooter(
        "マップ中の丸をクリックしても該当ページへ飛べます．",
        className="font-weight-light",
    )],
    id="link-card",
)


search_component = dbc.Col([
    dbc.Row([
        dbc.Col(
            id='search-form-div',
            children=dcc.Input(
                id='search-form',
                type="text",
                placeholder="検索ワードを入力してください",
                style=dict(width="100%"),
                className="input-control"),
            width=10,
        ),
        dbc.Col(
            html.Div(
                id='explore-start',
                children="検索！",
                className="btn btn-primary btn-lg",
            ),
            width=2,
        )],
        align="center")],
    style={"padding":"10px"},
    md=12,
    xl=6,
    className="card",
)

view_options = dbc.Col([
    dbc.Row(
        dbc.RadioItems(
            options=[
                {'label': 'SOM', 'value': 'SOM'},
                {'label': 'UKR', 'value': 'UKR'},
            ],
            value='TSOM',
            id="model-selector",
            style={'textAlign': "center", "display": "none"},
            className="h3",
        ),
    ),
    dbc.Row(
        dbc.RadioItems(
            options=[
                {'label': 'U-matrix 表示', 'value': 'U-matrix'},
                {'label': 'クラスタ表示', 'value': 'topic'},
            ],
            value='U-matrix',
            id="viewer-selector",
            inline=True,
            className="h3",
        ),
        style=dict(height="60%", width="100%", padding="10"),
        align="center",
    )],
    md=12,
    xl=6,
    style={"min-height":"100px", "padding-left":"30px"},
    className="card",
)


result_component = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                dcc.Graph(
                    id='example-graph',
                    figure=make_figure("Machine Learning", "TSOM", viewer_id="viewer_1"),
                    config=dict(displayModeBar=False)
                ),
                id="loading"
            ),
            style={"height": "100%",},
            md=12,
            xl=9,
            className="card",
        ),
        dbc.Col(
            dcc.Loading(
                dcc.Graph(
                    id='example-graph2',
                    figure=make_figure("Machine Learning", "TSOM", viewer_id="viewer_2"),
                    config=dict(displayModeBar=False)
                ),
                id="loading2"
            ),
            style={"height": "100%",},
            md=12,
            xl=9,
            className="card",
        ),
        dbc.Col(
            link_card,
            md=12,
            xl=3
        )
    ],
    align="center",
    className="h-75",
    style={"min-height": "70vh",},
    no_gutters=True
)


main_layout = dbc.Container(children=[
    dbc.Row([
        dbc.Col(
            html.H1(
                id='title',
                children='情報探索エンジン',
                className="display-2",
                style=dict(
                    fontFamily="Oswald, sans-serif"
                )
            ),
            md=12,
            xl=6
        ),
        dbc.Col(
            html.Div(
            children=[
                "情報探索をサポートする Web アプリケーションです．", html.Br(),
                "Google 検索結果を2次元にマッピングし，", html.Br(),
                "さらに勾配計算やクラスタリングをすることによって", html.Br(),
                "情報探索をサポートします．",
            ],
            className="h4"),
            md=12,
            xl=6
        )
    ], style={"min-height":"10vh", "margin-top":"10px"},
    align="end"),
    html.Hr(),
    # dbc.Button(
    #     "U-Matrix 表示とは？", id="open-umatrix-modal", className="ml-auto", n_clicks=0
    # ),
    umatrix_modal,
    dbc.Row([
        search_component,
        view_options
        ],
        style={"min-height":"10vh"}),
    result_component,
])


landing_page_layout = dbc.Container(children=[
    html.H1('Hello.'),
    search_component,
])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])
