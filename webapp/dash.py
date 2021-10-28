# -*- coding: utf-8 -*-

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from webapp import app
from webapp.figure_maker import make_first_figure


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


view_options = dbc.Col([
    dbc.Row(
        dbc.RadioItems(
            options=[
                {'label': 'U-matrix 表示', 'value': 'U-matrix'},
                {'label': 'CCP 表示', 'value': 'CCP'},
                {'label': 'クラスタ表示', 'value': 'topic'},
            ],
            value='U-matrix',
            id="viewer-selector",
            inline=True,
            className="h3",
        ),
        style=dict(height="50%", padding="10"),
        align="center",
    )],
    width=12,
    style={"padding-left":"30px", "height": "50%"},
    className="card",
)


make_search_component = lambda landing: dbc.Col([
    dbc.Row([
        dbc.Col(
            id=f'{"landing-" if landing else ""}search-form-div',
            children=dcc.Input(
                id=f'{"landing-" if landing else ""}search-form',
                type="text",
                placeholder="検索ワードを入力してください",
                style=dict(width="100%", fontSize="18px"),
                className="form-control form-control-lg"),
            width=10,
        ),
        dbc.Col(
            dbc.Button(
                id=f'{"landing-" if landing else ""}explore-start',
                children="検索！",
                color="primary",
                className="btn btn-primary btn-lg",
                style=dict(fontSize='0.8rem'),
            ),
            width=2,
        ),
        view_options if not landing else None,
        ],
        align="center")],
    style={"padding":"10px"},
    md=12,
    xl=8,
    className=f"card {'landing--search-form' if landing else ''}",
)


make_map = lambda id, viewer_id: dbc.Col(
    id=f'{id}-col',
    children=dcc.Loading(
        dcc.Graph(
            id=id,
            figure=make_first_figure(viewer_id),
            config=dict(displayModeBar=False),
        ),
        id=f'{id}-loading',
    ),
    style={"height": "100%", "display":"none"},
    md=12,
    xl=6,
    className="card",
)


result_component = dbc.Row(
    [
        make_map('paper-map', 'viewer_1'),
        make_map('word-map',  'viewer_2'),
    ],
    align="center",
    className="h-75",
    style={"min-height": "60vh",},
    no_gutters=True
)


word_addition_popover = dbc.Popover(
    id='word-addition-popover',
    target='paper-list-title',
    children=dbc.Button(
        id='word-addition-popover-button',
        children="検索！",
        className="btn btn-lg",
    ),
    trigger='focus',
    className='bg-secondary',
    style=dict(borderRight="#6c757d"),
)


paper_list = html.Div(
    id='paper-list',
    children=[
        dbc.Col(
            id='paper-list-title',
            children="",
            className="display-4",
            style=dict(
                fontFamily="Oswald, sans-serif",
                textAlign="center",
            ),
            width=dict(size=6, offset=3)
        ),
        word_addition_popover,
        html.Div(
            id='paper-list-components',
            children=[],
            style=dict(
                borderWidth="10px",
                borderColor="white",
                borderStyle="solid",
                borderRadius="5px",
            ),
        ),
    ]
)


main_layout = dbc.Container(children=[
    dbc.Row([
        dbc.Col(
            html.H1(
                id='title',
                children='論文探索エンジン',
                className="display-5",
                style=dict(
                    fontFamily="Oswald, sans-serif",
                    textAlign="center",
                )
            ),
            md=12,
            xl=4,
            align="center",
        ),
        make_search_component(landing=False),
        ],
    style={"min-height":"10vh", "margin-top":"10px"},
    align="center"),
    html.Hr(),
    # umatrix_modal,
    # dbc.Row([
    #     view_options
    #     ],
    #     style={"min-height":"5vh"}),
    result_component,
    html.Hr(),
    paper_list,
], id='main', style=dict(display="none"))


landing_page_layout = dbc.Container(
    id='landing',
    className='landing',
    children=[
        html.Div([
            html.Div(className='landing--box--green'),
            html.Div(className='landing--box--yellow'),
        ], className='landing--box'),
        html.H4(
            '論文探索エンジン',
            className="landing--title"
        ),
        html.Div(
            children=[
                "arXiv のデータベースと AI 技術を活用した", html.Br(),
                "論文探しをサポートする Web アプリケーションです．", html.Br(),
                "2つのマップと3種類の可視化方法で", html.Br(),
                "新しい論文探索体験を提供します．", html.Br(),
            ],
            className='landing--short-description',
        ),
        make_search_component(landing=True),
        # html.Div(
        #     '使い方はこちら',
        #     className='landing--howto-navi',
        # ),
])


app.layout = html.Div([
    landing_page_layout,
    main_layout,
    dcc.Store(id='memory'),
])
