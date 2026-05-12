// All 2026 FIFA World Cup Fixtures
const FIXTURES = {
  groups: {
    A: { teams: ["Mexico","South Korea","Czechia","South Africa"] },
    B: { teams: ["Canada","Qatar","Switzerland","Bosnia and Herzegovina"] },
    C: { teams: ["Brazil","Morocco","Scotland","Haiti"] },
    D: { teams: ["USA","Australia","Türkiye","Paraguay"] },
    E: { teams: ["Germany","Ivory Coast","Ecuador","Curaçao"] },
    F: { teams: ["Netherlands","Sweden","Japan","Tunisia"] },
    G: { teams: ["Belgium","Iran","Egypt","New Zealand"] },
    H: { teams: ["Spain","Saudi Arabia","Uruguay","Cabo Verde"] },
    I: { teams: ["France","Senegal","Norway","Iraq"] },
    J: { teams: ["Argentina","Austria","Algeria","Jordan"] },
    K: { teams: ["Portugal","Colombia","Uzbekistan","Congo DR"] },
    L: { teams: ["England","Croatia","Ghana","Panama"] }
  },
  matchdays: [
    {
      day: "MATCHDAY 1", dates: "June 11–17",
      matches: [
        { date:"Thu Jun 11", home:"Mexico", away:"South Africa", group:"A", venue:"Estadio Azteca, Mexico City" },
        { date:"Thu Jun 11", home:"South Korea", away:"Czechia", group:"A", venue:"Estadio Akron, Guadalajara" },
        { date:"Fri Jun 12", home:"Canada", away:"Bosnia and Herzegovina", group:"B", venue:"BMO Field, Toronto" },
        { date:"Fri Jun 12", home:"USA", away:"Paraguay", group:"D", venue:"SoFi Stadium, Los Angeles" },
        { date:"Sat Jun 13", home:"Qatar", away:"Switzerland", group:"B", venue:"Levi's Stadium, San Francisco" },
        { date:"Sat Jun 13", home:"Brazil", away:"Morocco", group:"C", venue:"MetLife Stadium, New York/NJ" },
        { date:"Sat Jun 13", home:"Haiti", away:"Scotland", group:"C", venue:"Gillette Stadium, Boston" },
        { date:"Sat Jun 13", home:"Australia", away:"Türkiye", group:"D", venue:"BC Place, Vancouver" },
        { date:"Sun Jun 14", home:"Ivory Coast", away:"Ecuador", group:"E", venue:"Lincoln Financial Field, Philadelphia" },
        { date:"Sun Jun 14", home:"Germany", away:"Curaçao", group:"E", venue:"NRG Stadium, Houston" },
        { date:"Sun Jun 14", home:"Netherlands", away:"Japan", group:"F", venue:"AT&T Stadium, Dallas" },
        { date:"Sun Jun 14", home:"Sweden", away:"Tunisia", group:"F", venue:"Estadio BBVA, Monterrey" },
        { date:"Mon Jun 15", home:"Saudi Arabia", away:"Uruguay", group:"H", venue:"Hard Rock Stadium, Miami" },
        { date:"Mon Jun 15", home:"Spain", away:"Cabo Verde", group:"H", venue:"Mercedes-Benz Stadium, Atlanta" },
        { date:"Mon Jun 15", home:"Iran", away:"New Zealand", group:"G", venue:"SoFi Stadium, Los Angeles" },
        { date:"Mon Jun 15", home:"Belgium", away:"Egypt", group:"G", venue:"Lumen Field, Seattle" },
        { date:"Tue Jun 16", home:"France", away:"Senegal", group:"I", venue:"MetLife Stadium, New York/NJ" },
        { date:"Tue Jun 16", home:"Iraq", away:"Norway", group:"I", venue:"Gillette Stadium, Boston" },
        { date:"Tue Jun 16", home:"Argentina", away:"Algeria", group:"J", venue:"Arrowhead Stadium, Kansas City" },
        { date:"Tue Jun 16", home:"Austria", away:"Jordan", group:"J", venue:"Levi's Stadium, San Francisco" },
        { date:"Wed Jun 17", home:"Ghana", away:"Panama", group:"L", venue:"BMO Field, Toronto" },
        { date:"Wed Jun 17", home:"England", away:"Croatia", group:"L", venue:"AT&T Stadium, Dallas" },
        { date:"Wed Jun 17", home:"Portugal", away:"Congo DR", group:"K", venue:"NRG Stadium, Houston" },
        { date:"Wed Jun 17", home:"Uzbekistan", away:"Colombia", group:"K", venue:"Estadio Azteca, Mexico City" }
      ]
    },
    {
      day: "MATCHDAY 2", dates: "June 18–23",
      matches: [
        { date:"Thu Jun 18", home:"Czechia", away:"South Africa", group:"A", venue:"Mercedes-Benz Stadium, Atlanta" },
        { date:"Thu Jun 18", home:"Switzerland", away:"Bosnia and Herzegovina", group:"B", venue:"SoFi Stadium, Los Angeles" },
        { date:"Thu Jun 18", home:"Canada", away:"Qatar", group:"B", venue:"BC Place, Vancouver" },
        { date:"Thu Jun 18", home:"Mexico", away:"South Korea", group:"A", venue:"Estadio Akron, Guadalajara" },
        { date:"Fri Jun 19", home:"Brazil", away:"Haiti", group:"C", venue:"Lincoln Financial Field, Philadelphia" },
        { date:"Fri Jun 19", home:"Scotland", away:"Morocco", group:"C", venue:"Gillette Stadium, Boston" },
        { date:"Fri Jun 19", home:"Türkiye", away:"Paraguay", group:"D", venue:"Levi's Stadium, San Francisco" },
        { date:"Fri Jun 19", home:"USA", away:"Australia", group:"D", venue:"Lumen Field, Seattle" },
        { date:"Sat Jun 20", home:"Germany", away:"Ivory Coast", group:"E", venue:"BMO Field, Toronto" },
        { date:"Sat Jun 20", home:"Ecuador", away:"Curaçao", group:"E", venue:"Arrowhead Stadium, Kansas City" },
        { date:"Sat Jun 20", home:"Netherlands", away:"Sweden", group:"F", venue:"NRG Stadium, Houston" },
        { date:"Sat Jun 20", home:"Tunisia", away:"Japan", group:"F", venue:"Estadio BBVA, Monterrey" },
        { date:"Sun Jun 21", home:"Uruguay", away:"Cabo Verde", group:"H", venue:"Hard Rock Stadium, Miami" },
        { date:"Sun Jun 21", home:"Spain", away:"Saudi Arabia", group:"H", venue:"Mercedes-Benz Stadium, Atlanta" },
        { date:"Sun Jun 21", home:"Belgium", away:"Iran", group:"G", venue:"SoFi Stadium, Los Angeles" },
        { date:"Sun Jun 21", home:"New Zealand", away:"Egypt", group:"G", venue:"BC Place, Vancouver" },
        { date:"Mon Jun 22", home:"Norway", away:"Senegal", group:"I", venue:"MetLife Stadium, New York/NJ" },
        { date:"Mon Jun 22", home:"France", away:"Iraq", group:"I", venue:"Lincoln Financial Field, Philadelphia" },
        { date:"Mon Jun 22", home:"Argentina", away:"Austria", group:"J", venue:"AT&T Stadium, Dallas" },
        { date:"Mon Jun 22", home:"Jordan", away:"Algeria", group:"J", venue:"Levi's Stadium, San Francisco" },
        { date:"Tue Jun 23", home:"England", away:"Ghana", group:"L", venue:"Gillette Stadium, Boston" },
        { date:"Tue Jun 23", home:"Panama", away:"Croatia", group:"L", venue:"BMO Field, Toronto" },
        { date:"Tue Jun 23", home:"Portugal", away:"Uzbekistan", group:"K", venue:"NRG Stadium, Houston" },
        { date:"Tue Jun 23", home:"Colombia", away:"Congo DR", group:"K", venue:"Estadio Akron, Guadalajara" }
      ]
    },
    {
      day: "MATCHDAY 3", dates: "June 24–27 (Simultaneous)",
      matches: [
        { date:"Wed Jun 24", home:"Switzerland", away:"Canada", group:"B", venue:"BC Place, Vancouver" },
        { date:"Wed Jun 24", home:"Bosnia and Herzegovina", away:"Qatar", group:"B", venue:"Lumen Field, Seattle" },
        { date:"Wed Jun 24", home:"Scotland", away:"Brazil", group:"C", venue:"Hard Rock Stadium, Miami" },
        { date:"Wed Jun 24", home:"Morocco", away:"Haiti", group:"C", venue:"Mercedes-Benz Stadium, Atlanta" },
        { date:"Wed Jun 24", home:"Czechia", away:"Mexico", group:"A", venue:"Estadio Azteca, Mexico City" },
        { date:"Wed Jun 24", home:"South Africa", away:"South Korea", group:"A", venue:"Estadio BBVA, Monterrey" },
        { date:"Thu Jun 25", home:"Ecuador", away:"Germany", group:"E", venue:"MetLife Stadium, New York/NJ" },
        { date:"Thu Jun 25", home:"Curaçao", away:"Ivory Coast", group:"E", venue:"Lincoln Financial Field, Philadelphia" },
        { date:"Thu Jun 25", home:"Japan", away:"Sweden", group:"F", venue:"AT&T Stadium, Dallas" },
        { date:"Thu Jun 25", home:"Tunisia", away:"Netherlands", group:"F", venue:"Arrowhead Stadium, Kansas City" },
        { date:"Thu Jun 25", home:"Türkiye", away:"USA", group:"D", venue:"SoFi Stadium, Los Angeles" },
        { date:"Thu Jun 25", home:"Paraguay", away:"Australia", group:"D", venue:"Levi's Stadium, San Francisco" },
        { date:"Fri Jun 26", home:"Norway", away:"France", group:"I", venue:"Gillette Stadium, Boston" },
        { date:"Fri Jun 26", home:"Senegal", away:"Iraq", group:"I", venue:"Lincoln Financial Field, Philadelphia" },
        { date:"Fri Jun 26", home:"Algeria", away:"Austria", group:"J", venue:"Levi's Stadium, San Francisco" },
        { date:"Fri Jun 26", home:"Jordan", away:"Argentina", group:"J", venue:"AT&T Stadium, Dallas" },
        { date:"Sat Jun 27", home:"Congo DR", away:"Uzbekistan", group:"K", venue:"Mercedes-Benz Stadium, Atlanta" },
        { date:"Sat Jun 27", home:"Colombia", away:"Portugal", group:"K", venue:"NRG Stadium, Houston" },
        { date:"Sat Jun 27", home:"Croatia", away:"Ghana", group:"L", venue:"BMO Field, Toronto" },
        { date:"Sat Jun 27", home:"Panama", away:"England", group:"L", venue:"MetLife Stadium, New York/NJ" }
      ]
    }
  ],
  knockout: [
    { round: "ROUND OF 32", dates: "Jun 28 – Jul 3", venue: "All Host Cities", icon: "⚔️" },
    { round: "ROUND OF 16", dates: "Jul 4 – Jul 7", venue: "Philadelphia, Houston, MetLife, Dallas, Atlanta, Seattle, Vancouver, Mexico City", icon: "🔥" },
    { round: "QUARTER-FINALS", dates: "Jul 9 – Jul 11", venue: "Boston, Los Angeles, Miami, Kansas City", icon: "⚡" },
    { round: "SEMI-FINALS", dates: "Jul 14–15", venue: "AT&T Stadium Dallas & Mercedes-Benz Atlanta", icon: "👑" },
    { round: "THIRD PLACE", dates: "Sat Jul 18", venue: "Hard Rock Stadium, Miami", icon: "🥉" },
    { round: "THE FINAL 🏆", dates: "Sun Jul 19", venue: "MetLife Stadium, New York/New Jersey", icon: "🏆" }
  ]
};
