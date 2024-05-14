let summer = {
  data: ["Jun", "Jul", "Aug"],
};

let winter = {
  data: ["Dec", "Jan", "Feb"],
  getLogger: function () {
    return () => console.log(this.data);
  },
};

var logger = winter.getLogger();
logger();
logger.call(summer);
