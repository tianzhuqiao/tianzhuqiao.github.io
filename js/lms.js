function plot_lms(id, mu=0.1, target=10, initial=0, awgn_std=1, num=200, delay=0) {
    var x = [], y = [];
    /*function to plot*/
    points=[];
    var correction = Array(delay+1).fill(0)
    var current = initial;
    for (i=0;i<=num;i++)
    {
        x.push(i);
        y.push(current);
        var estimation = current + randn_bm() * awgn_std;
        var error = target - estimation;
        correction.shift()
        correction.push(mu*error)
        current = current + correction[0];
    }


	TESTER = document.getElementById(id);

    Plotly.newPlot( TESTER, [{
        x: x,
        y: y,
        name: "value"}, {
            x: [0, num],
            y: [target, target],
            name: "target"
        }],
        {
            margin: { t: 20, b: 20 }
        }
    );
}
function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function plot_lms_variance(id, awgn_std=1) {
    var x = [], v0 = [], v1 = [];
    for (i=0;i<70;i++)
    {
        var mu = i/100;

        x.push(mu);
        v0.push(mu/(2-mu));
        if (mu == 0) {
            v1.push(0);
        } else {
            v1.push((1+mu)*mu**2/(1-mu)/((1+mu)**2-1));
        }
    }


    TESTER = document.getElementById(id);

    Plotly.newPlot( TESTER, [{
        x: x,
        y: v0,
        name: "no delay"}, {
            x: x,
            y: v1,
            name: "1 delay"
        }],
        {
            margin: { t: 50, b: 50 },
            xaxis: {title: 'mu'},
            yaxis: {title: "var(v)"},
            title: "variance of signal when converges (var(n) = 1)",
        }
    );
}

function plot_lms_group_delay(id) {
    var x = [], v0 = [], v1 = [];
    for (i=1;i<50;i++)
    {
        var mu = i/100;

        x.push(mu);
        v0.push((1-mu)/mu);
        v1.push((1-2*mu)/mu);
    }


    TESTER = document.getElementById(id);

    Plotly.newPlot( TESTER, [{
        x: x,
        y: v0,
        name: "no delay"}, {
            x: x,
            y: v1,
            name: "1 delay"
        }],
        {
            margin: { t: 50, b: 50 },
            xaxis: {title: 'mu'},
            yaxis: {title: "group delay"},
            title: "group delay at frequency 0",
        }
    );
}

