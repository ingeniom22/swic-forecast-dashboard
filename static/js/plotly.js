document.addEventListener('DOMContentLoaded', function () {
    var chart_data = {{ chart | safe; }};
    // var histogramData = {{ bfs_histogram | safe }};
    // var appBarData = {{ app_bar | safe }};
    var config = { responsive: true };

    Plotly.newPlot('chart', chart_data.data, chart_data.layout, config);
    // Plotly.newPlot('bfsHistogram', histogramData.data, histogramData.layout, config);
    // Plotly.newPlot('appBar', appBarData.data, appBarData.layout, config);

    var d3 = Plotly.d3;
    var WIDTH_IN_PERCENT_OF_PARENT = 100,
        HEIGHT_IN_PERCENT_OF_PARENT = 100;

    var gd3 = d3.selectAll(".responsive-plot")
        .style({
            width: '90%',
            'margin-left': (100 - WIDTH_IN_PERCENT_OF_PARENT) / 2 + '%',

            height: 'fit-content',
            'margin-top': '0vh'
        });

    var nodes_to_resize = gd3[0];
    window.onresize = function () {
        for (var i = 0; nodes_to_resize && i < nodes_to_resize.length; i++) {
            Plotly.Plots.resize(nodes_to_resize[i]);
        }
    };
});