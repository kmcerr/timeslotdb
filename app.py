import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
TIME_SLOTS = [
    '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
    '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30'
]


class DataProcessor:
    @staticmethod
    def load_data(file_path):
        try:
            # Load data
            data = pd.read_csv(file_path)
            logging.info(f"Data loaded. Shape: {data.shape}")

            # Process dates
            data['date'] = pd.to_datetime(data['date'])
            data['month_year'] = data['date'].dt.to_period('M').astype(str)
            data['month'] = data['date'].dt.strftime('%b')
            data['year'] = data['date'].dt.year
            data['day'] = data['date'].dt.day_name()

            # Process time
            data['time'] = pd.to_datetime(
                data['time'], format='%H:%M:%S', errors='coerce'
            ).dt.strftime('%H:%M')

            # Make month categorical
            data['month'] = pd.Categorical(
                data['month'], categories=MONTHS, ordered=True
            )

            # Standardize received column
            received_mapping = {
                1: 'Received', 0: 'Not Received',
                '1': 'Received', '0': 'Not Received',
                'received': 'Received', 'not_received': 'Not Received'
            }
            data['received'] = data['received'].map(received_mapping)

            logging.info(f"Unique values in 'received': {data['received'].unique()}")

            return data.sort_values(by=['year', 'month', 'time'])

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def calculate_percentages(grouped_data):
        try:
            total = grouped_data.sum(axis=1)
            percentages = grouped_data.div(total, axis=0) * 100
            return percentages.round(2)
        except Exception as e:
            logging.error(f"Error calculating percentages: {str(e)}")
            return pd.DataFrame()


class GraphBuilder:
    @staticmethod
    def create_trend_figure(x_values, y_values_dict, title, xaxis_title, yaxis_title,
                            show_options, category_order=None, layout_updates=None):
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly * (
                (len(y_values_dict) // len(px.colors.qualitative.Plotly)) + 1
        )

        for (label, y_values), color in zip(y_values_dict.items(), colors):
            show_numbers = 'numbers' in show_options
            text = y_values.apply(lambda x: f'{x:.2f}%') if show_numbers else None

            # Add bar chart
            if 'bar' in show_options:
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=y_values,
                    name=label,
                    marker_color=color,
                    text=text,
                    textposition='auto' if show_numbers else 'none',
                    hovertemplate=f'<b>{label}</b><br>{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:.2f}}%<extra></extra>'
                ))

            # Add scatter plot
            if 'distribution' in show_options:
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers+text' if show_numbers else 'lines+markers',
                    name=f'Scatter ({label})',
                    line=dict(color=color),
                    marker=dict(symbol='circle', size=8),
                    text=text if show_numbers else None,
                    textposition='top center',
                    showlegend=True,
                    hovertemplate=f'<b>{label}</b><br>{xaxis_title}: %{{x}}<br>{yaxis_title}: %{{y:.2f}}%<extra></extra>'
                ))

            # Add trendline
            if 'trendline' in show_options:
                GraphBuilder._add_trendline(fig, x_values, y_values, label, color)

        # Update layout
        layout_config = {
            'title': dict(text=title, y=0.95, x=0.5, xanchor='center', yanchor='top'),
            'xaxis_title': xaxis_title,
            'yaxis_title': yaxis_title,
            'yaxis': dict(range=[0, 100], tickformat='.2f'),
            'hovermode': 'x unified',
            'margin': dict(t=100)
        }

        # Add category order if specified
        if category_order:
            layout_config['xaxis'] = {'categoryorder': 'array', 'categoryarray': category_order}

        # Apply any additional layout updates
        if layout_updates:
            layout_config.update(layout_updates)

        fig.update_layout(**layout_config)

        return fig

    @staticmethod
    def _add_trendline(fig, x_values, y_values, label, color):
        valid_mask = y_values.notnull() & (y_values != 0)
        if valid_mask.sum() >= 2:
            x_valid = [x for x, valid in zip(x_values, valid_mask) if valid]
            x_numeric = [x_values.index(x) for x in x_valid]
            y_valid = y_values[valid_mask].values

            z = np.polyfit(x_numeric, y_valid, 1)
            p = np.poly1d(z)
            trendline_y = p(x_numeric)

            fig.add_trace(go.Scatter(
                x=x_valid,
                y=trendline_y,
                mode='lines',
                name=f'Trend ({label})',
                line=dict(color=color, dash='dot'),
                hoverinfo='skip'
            ))


class DashboardApp:
    def __init__(self, data):
        self.data = data
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        self.app.layout = html.Div([
            # Header Section 1
            html.H2('Projects Received and Not Received Trends by Time Slot'),

            # Controls Row 1
            html.Div([
                # Time Aggregation Dropdown
                html.Div([
                    html.Label('Select Time Aggregation'),
                    dcc.Dropdown(
                        id='time-aggregation',
                        options=[
                            {'label': 'Month-Year', 'value': 'month_year'},
                            {'label': 'Month Only', 'value': 'month'},
                            {'label': 'Year Only', 'value': 'year'},
                            {'label': 'Day', 'value': 'day'}
                        ],
                        value='month_year'
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),

                # Period Filter Dropdown
                html.Div([
                    html.Label('Select Specific Periods'),
                    dcc.Dropdown(id='period-filter', multi=True)
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),

            # Controls Row 2
            html.Div([
                # Day Selector
                html.Div([
                    html.Label('Select Days of the Week'),
                    dcc.Dropdown(
                        id='day-of-week-selector',
                        options=[{'label': day, 'value': day} for day in DAYS_OF_WEEK],
                        multi=True,
                        value=DAYS_OF_WEEK
                    )
                ], id='day-selector-container',
                    style={'width': '48%', 'display': 'inline-block', 'margin-top': '10px'}),

                # Received Status Selector
                html.Div([
                    html.Label('Select Received Status'),
                    dcc.Dropdown(
                        id='received-status-selector',
                        options=[
                            {'label': 'Received', 'value': 'Received'},
                            {'label': 'Not Received', 'value': 'Not Received'},
                            {'label': 'Both', 'value': 'Both'}
                        ],
                        value='Both'
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'margin-top': '10px'}),
            ]),

            # Graph Options 1
            html.Div([
                dcc.Checklist(
                    id='trend-graph-options',
                    options=[
                        {'label': 'Show Bar Graph', 'value': 'bar'},
                        {'label': 'Show Trendline', 'value': 'trendline'},
                        {'label': 'Show Scatter Plot', 'value': 'distribution'},
                        {'label': 'Show Numerical Values', 'value': 'numbers'}
                    ],
                    value=['bar'],
                    inline=True
                ),
            ], style={'margin-top': '10px'}),

            # Graph 1
            dcc.Graph(id='trend-graph'),

            # Header Section 2
            html.H2('Comparison of Received and Not Received Projects by Month and Year',
                    style={'margin-top': '40px'}),

            # Controls Row 3
            html.Div([
                # Year Selector
                html.Div([
                    html.Label('Select Years'),
                    dcc.Dropdown(
                        id='unified-year-selector',
                        options=[{'label': str(year), 'value': year}
                                 for year in sorted(self.data['year'].unique())],
                        multi=True,
                        value=sorted(self.data['year'].unique())
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),

                # Month Selector
                html.Div([
                    html.Label('Select Months'),
                    dcc.Dropdown(
                        id='unified-month-selector',
                        options=[{'label': month, 'value': month} for month in MONTHS],
                        multi=True,
                        value=MONTHS
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),

            # Received Status Selector 2
            html.Div([
                html.Label('Select Received Status'),
                dcc.Dropdown(
                    id='unified-received-status-selector',
                    options=[
                        {'label': 'Received', 'value': 'Received'},
                        {'label': 'Not Received', 'value': 'Not Received'},
                        {'label': 'Both', 'value': 'Both'}
                    ],
                    value='Both'
                )
            ], style={'width': '48%', 'display': 'inline-block', 'margin-top': '10px'}),

            # Graph Options 2
            html.Div([
                dcc.Checklist(
                    id='unified-graph-options',
                    options=[
                        {'label': 'Show Bar Graph', 'value': 'bar'},
                        {'label': 'Show Trendline', 'value': 'trendline'},
                        {'label': 'Show Scatter Plot', 'value': 'distribution'},
                        {'label': 'Show Numerical Values', 'value': 'numbers'}
                    ],
                    value=['bar'],
                    inline=True
                ),
            ], style={'margin-top': '10px'}),

            # Graph 2
            dcc.Graph(id='unified-trend-graph'),
        ])

    def _setup_callbacks(self):
        @self.app.callback(
            [Output('period-filter', 'options'), Output('period-filter', 'value')],
            [Input('time-aggregation', 'value')]
        )
        def update_period_filter(aggregation_type):
            options_map = {
                'month_year': sorted(self.data['month_year'].unique()),
                'year': sorted(self.data['year'].unique()),
                'month': self.data['month'].cat.categories.tolist(),
                'day': DAYS_OF_WEEK
            }

            values = options_map.get(aggregation_type, [])
            options = [{'label': str(v), 'value': v} for v in values]
            value = [options[-1]['value']] if options else []

            return options, value

        @self.app.callback(
            Output('day-selector-container', 'style'),
            [Input('time-aggregation', 'value')]
        )
        def update_day_selector_visibility(aggregation_type):
            """Hide day selector when aggregation is by day"""
            if aggregation_type == 'day':
                return {'display': 'none'}
            return {'width': '48%', 'display': 'inline-block', 'margin-top': '10px'}

        @self.app.callback(
            Output('trend-graph', 'figure'),
            [Input('time-aggregation', 'value'),
             Input('period-filter', 'value'),
             Input('day-of-week-selector', 'value'),
             Input('received-status-selector', 'value'),
             Input('trend-graph-options', 'value')]
        )
        def update_trend_graph(aggregation_type, selected_periods, selected_days,
                               received_status, show_options):
            if not selected_periods:
                return go.Figure()

            y_values_dict = {}
            received_statuses = ['Received', 'Not Received'] if received_status == 'Both' else [received_status]

            for period in selected_periods:
                # Filter data based on aggregation type
                filtered_data = self._filter_data_by_period(aggregation_type, period)

                if filtered_data.empty:
                    continue

                # Apply day filter if needed
                if selected_days and aggregation_type != 'day':
                    filtered_data = filtered_data[filtered_data['day'].isin(selected_days)]

                # Calculate percentages
                trend_data = filtered_data.groupby(['time', 'received']).size().unstack(fill_value=0)
                trend_data_percent = DataProcessor.calculate_percentages(trend_data)
                trend_data_percent = trend_data_percent.reindex(TIME_SLOTS, fill_value=0)

                # Build y_values for each status
                for status in received_statuses:
                    y_values = trend_data_percent.get(status, pd.Series(0, index=TIME_SLOTS))
                    label = f'{period} - {status}'
                    y_values_dict[label] = y_values

            # Build title
            title = self._build_trend_title(aggregation_type, selected_periods, selected_days)

            return GraphBuilder.create_trend_figure(
                TIME_SLOTS, y_values_dict, title, 'Time Slot', 'Percentage',
                show_options, category_order=TIME_SLOTS
            )

        @self.app.callback(
            Output('unified-trend-graph', 'figure'),
            [Input('unified-year-selector', 'value'),
             Input('unified-month-selector', 'value'),
             Input('unified-received-status-selector', 'value'),
             Input('unified-graph-options', 'value')]
        )
        def update_unified_trend_graph(selected_years, selected_months,
                                       received_status, show_options):
            if not selected_years or not selected_months:
                return go.Figure()

            y_values_dict = {}
            received_statuses = ['Received', 'Not Received'] if received_status == 'Both' else [received_status]

            for year in selected_years:
                for status in received_statuses:
                    # Filter data
                    filtered_data = self.data[
                        (self.data['year'] == year) &
                        (self.data['month'].isin(selected_months)) &
                        (self.data['received'] == status)
                        ]

                    if filtered_data.empty:
                        continue

                    # Calculate percentages by month
                    trend_data = filtered_data.groupby('month').size().reindex(MONTHS, fill_value=0)
                    total_projects = self.data[
                        (self.data['year'] == year) &
                        (self.data['month'].isin(selected_months))
                        ].groupby('month').size().reindex(MONTHS, fill_value=0)

                    y_values = (trend_data / total_projects) * 100
                    y_values = y_values.fillna(0).round(2)

                    label = f'{year} - {status}'
                    y_values_dict[label] = y_values

            # Build title
            title = (f'Comparison of Projects by Month and Year<br>'
                     f'Selected Years: {", ".join(map(str, selected_years))}<br>'
                     f'Selected Months: {", ".join(selected_months)}')

            return GraphBuilder.create_trend_figure(
                MONTHS, y_values_dict, title, 'Month', 'Percentage',
                show_options, category_order=MONTHS,
                layout_updates={'barmode': 'group'}
            )

    def _filter_data_by_period(self, aggregation_type, period):
        filters = {
            'month_year': self.data['month_year'] == period,
            'year': self.data['year'] == period,
            'month': self.data['month'] == period,
            'day': self.data['day'] == period
        }

        if aggregation_type in filters:
            return self.data[filters[aggregation_type]]
        return self.data.copy()

    def _build_trend_title(self, aggregation_type, selected_periods, selected_days):
        period_labels = {
            'month_year': 'Months-Years',
            'year': 'Years',
            'month': 'Months',
            'day': 'Days'
        }

        title = (f'Distribution of Projects by Time Slot<br>'
                 f'Selected {period_labels.get(aggregation_type, "Periods")}: '
                 f'{", ".join(map(str, selected_periods))}')

        if selected_days and aggregation_type != 'day':
            title += f'<br>Selected Days: {", ".join(selected_days)}'
        elif aggregation_type != 'day':
            title += '<br>Selected Days: All'

        return title

    def run(self, **kwargs):
        self.app.run(**kwargs)


if __name__ == '__main__':
    file_path = 'sample_data/sample_data.csv'
    data = DataProcessor.load_data(file_path)

    if data.empty:
        logging.error("Please check the file path.")
    else:
        dashboard = DashboardApp(data)
        dashboard.run(port=8051, debug=True)