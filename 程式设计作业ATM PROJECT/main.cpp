#include<iostream>
using namespace std;
#include<string>
#include<vector>
#include<sstream>
#include<fstream>
#include <ctime>
#include <thread>


//定义类和成员函数
class Person {
public:
	static bool judgeacc(string& account);
	bool judgepas01(string& password);
	bool judgepas02( string& password, int count01=3);
	bool getcash(double& get, double& dayout, double& daycash, double singlecash = 2000);
	bool changepassword(string& p1, string& p2);
	bool transfercash(string& t1, string& t2);
	bool gettrans(double& get, double& daytransout, double& daytrans, double singlecash = 2000);
	void showbalance();
	void getcash(double& dayout, double& daycash);
	void changepassword();
	void exitcard();
	void transfercash(double& daytransout, double& daytrans);
	void save();
	void login();
	int count = 3;

private:
	
	string password;
	double balance=0.0;
	string isfreeze = "1";
	string ID;
	string name;
	string IDcard;
	
	
};

//实现时钟的显示
void showClock() {
	while (true) {
		system("cls"); // Windows
		// system("clear"); // Linux / macOS

		time_t now = time(nullptr);
		struct tm timeinfo;
		localtime_s(&timeinfo, &now);

		cout << "当前时间: "
			<< timeinfo.tm_year + 1900 << "-"
			<< timeinfo.tm_mon + 1 << "-"
			<< timeinfo.tm_mday << " "
			<< timeinfo.tm_hour << ":"
			<< timeinfo.tm_min << ":"
			<< timeinfo.tm_sec << endl;

		this_thread::sleep_for(chrono::seconds(1));
	}
}

//欢迎界面
void welcomevision() {
	cout << "____________________" << endl;
	cout << "欢迎来到汇丰银行" << endl;
	cout << endl;
	cout << endl;
	cout << "请输入您的19位数字银行卡号" << endl;
	/*cout << "请输入您的6位数字卡号密码" << endl;*/
	cout << endl;
	cout << endl;
	cout << "祝您拥有美好的一天" << endl;
	cout << "____________________" << endl;
	
}

//功能界面
void functionvision() {
	cout << "----------------" << endl;
	cout << "欢迎来到功能界面" << endl;
	cout << endl;
	cout << "按1查询余额" << endl;
	cout << "按2进行取款" << endl;
	cout << "按3进行转账" << endl;
	cout << "按4修改密码" << endl;
	cout << "按5进行退卡" << endl;
	cout << endl;
	cout << "祝您拥有美好的一天" << endl;
	cout << "----------------" << endl;
	
}

//数据文件的保存
void Person::save()
{
	string path = "D:\\ATM PROJECT\\ATM\\user data\\";
	string temp = path + ID + ".txt";
	ofstream ofs(temp);
	ofs << "account01:" << ID << endl;
	ofs << "name01:" << name << endl;
	ofs << "idcard01:" << IDcard << endl;
	ofs << "password01:" << password << endl;
	ofs << "balance01:" << balance << endl;
	ofs << "isfreeze01:" << isfreeze << endl;
	ofs.close();
}

//判断账号的合法性
bool Person::judgeacc(string& account) {
	if (account.size() != 19) {
		cerr << "卡号输入格式有误，请重新输入" << endl;
		
		return true;
	}
	for (int i = 0; i < account.size(); i++)
	{

		if (account[i] < '0' || account[i] > '9')
		{
			return true;//如果账户有误，返回true重新进入循环输入卡号
		}
	}
		return false;
	
}

//实现登录功能
void Person::login() {
	_RELOGIN:
	string number;
	cin >> number;
	while (Person::judgeacc(number)) {
		cin >> number;
	}
	ID = number;
	string path = "D:\\ATM PROJECT\\ATM\\user data\\";
	string temp = path + ID + ".txt";
	ifstream ifs;
	ifs.open(temp, ios::in);
	if (!ifs.is_open()) {
		std::cout << "此账户不存在，请重新输入" << endl;
		goto _RELOGIN;
	}
	
	else {
		string account01, name01, idcard01, password01, balance01, dayout01, isfreeze01, temp;
		while (ifs >> temp) {
			if (temp.find("account01:") == 0) account01 = temp.substr(10); // 从冒号后截取内容
			else if (temp.find("name01:") == 0) name01 = temp.substr(7);
			else if (temp.find("idcard01:") == 0) idcard01 = temp.substr(9);
			else if (temp.find("password01:") == 0) password01 = temp.substr(11);
			else if (temp.find("balance01:") == 0) balance01 = temp.substr(10);
			else if (temp.find("isfreeze01:") == 0) isfreeze01 = temp.substr(11);
		}

		ID = account01;
		name = name01;
		IDcard = idcard01;
		password = password01;
		stringstream ss;
		ss << balance01;
		ss >> balance;
		isfreeze = isfreeze01;

		
		
		
		
		
		
		
		/*string str[7];
		for (int i = 0; i < 7; i++)
		{
			ifs >> str[i];
		}
		ID = str[0].substr(7);
		name = str[1].substr(7);
		IDcard = str[2].substr(10);
		password = str[3].substr(7);
		string balance01;
		balance01 = str[4].substr(7);
		stringstream ss;
		ss << balance01;
		ss >> balance;
		dayout = str[5].substr(19);
		isfreeze = str[6].substr(19);*/
	}
	ifs.close();
	if (isfreeze == "0"){
		cout<<"此账户已被冻结"<<endl;
		system("pause");
		std::exit(0);
	}
	cout << "请输入密码" << endl;

	string pas;
	cin >> pas;
	while (judgepas01(pas)) {
		cin >> pas;
	}
	while (judgepas02(pas,3)) {
		cin >> pas;
	}
	


}

//判断密码的合法性
bool Person::judgepas01(string& password) {
	if (password.size() != 6) {
		cerr << "密码输入格式有误，请重新输入" << endl;
		return true;
	}
	else {
		return false;
	}
}

bool Person::judgepas02(string& passwordin,int count01) {
	
	if (count ==1) {
		cout << "超出错误次数，该卡将被冻结" << endl;
		isfreeze = "0";
		save();
		std::exit(0);
	}
	if (passwordin == password) {
		cout << "登入成功" << endl;
		return false;
	}
	else {
		count--;
		cout << "验证错误  " << "请重新输入密码" << endl;
			cout<< "您还有" << count << "次输入机会" << endl;
		return true;
	}

}

//实现查询余额功能
void Person::showbalance() {
	cout << "您目前的余额为" <<balance << endl;
}

//判断取款金额的合法性
bool Person::getcash(double &get, double& dayout, double& daycash,double singlecash) {
	if (get > singlecash) {
		cout << "超出单次取款金额，请重新输入取款金额" << endl;
		return true;
	}
	if ((dayout + get) > daycash) {
		cout << "超出单日取款金额，请重新输入取款金额" << endl;
		return true;
	}
	if (get - (int)get> 0 && get - (int)get  < 1) {
		cout << "取款金额应为100的整数倍，请重新输入取款金额" << endl;
		return true;
	}
	if ((int)get % 100 != 0) {
		cout << "取款金额应为100的整数倍，请重新输入取款金额" << endl;
		return true;
	}
	if (get > balance) {
		cout << "余额不足，请重新输入取款金额" << endl;
		return true;
	}
	return false;
}

//实现取款功能
void Person::getcash(double &dayout,double&daycash) {
	
	cout << "请输入您的取款金额" << endl;
	double amount;
	cin >> amount;
	
	while (getcash(amount,dayout,daycash)) {
		cin >> amount;
	}
	cout << "取款成功" << endl;
	balance -= amount;
	dayout += amount;
	if (dayout == daycash) {
		cout << "今日取款限额已用尽" << endl;
	}
	else {
		cout << "今日剩余取款限额 " << daycash - dayout<< endl;
	}
	save();
	showbalance();

}

//判断修改密码的合法性
bool Person::changepassword(string& p1, string& p2) {
	if (p1 != p2) {
		cout << "密码输入不一致，请重新输入" << endl;
		return true;
	}
	return false;
}

//实现修改密码的功能
void Person::changepassword() {
	string newpassword01;
	string newpassword02;
	cout << "请输入6位新密码" << endl;
	cin >> newpassword01;
	cout << "请再次输入新密码" << endl;
	cin >> newpassword02;
	while (changepassword(newpassword01, newpassword02)) {
		cout << "请输入6位新密码" << endl;
		cin >> newpassword01;
		cout << "请再次输入新密码" << endl;
		cin >> newpassword02;
	}
	cout << "修改密码成功,请重新登录" << endl;
	password = newpassword01;
	save();
}

//实现退卡的功能
void Person::exitcard() {
	cout << "退卡成功！感谢使用本系统，再见！" << endl;
}

//判断转账账户的合法性
bool Person::transfercash(string& t1, string& t2) {
	if (t1.size() != 19 || t2.size() != 19) {
		cout << "输入转入账号有误，请重新输入" << endl;
		return true;
	}
	if (t1 != t2) {
		cout << "转入账号输入不一致，请重新输入" << endl;
		return true;
	}
	return false;
}

bool Person::gettrans(double& get, double& daytransout, double& daytrans, double singlecash) {
	if (get > singlecash) {
		cout << "超出单次转账金额，请重新输入转账金额" << endl;
		return true;
	}
	if ((daytransout + get) > daytrans) {
		cout << "超出单日转账金额，请重新输入转账金额" << endl;
		return true;
	}
	if ((int)get % 100 != 0) {
		cout << "转账金额应为100的整数倍，请重新输入转账金额" << endl;
		return true;
	}
	if (get >balance) {
		cout << "余额不足，请重新输入转账金额" << endl;
		return true;
	}
	return false;
}

//实现转账功能
void Person::transfercash(double& daytransout, double& daytrans) {
	string taccount01;
	string taccount02;
	cout << "请输入19位转入账号" << endl;
	cin >> taccount01;
	cout << "请再次输入转入账号" << endl;
	cin >> taccount02;
	while (transfercash(taccount01, taccount02)) {
		cout << "请输入19位转入账号" << endl;
		cin >> taccount01;
		cout << "请再次输入转入账号" << endl;
		cin >> taccount02;
	}

	cout << "请输入您的转账金额" << endl;
	double transamount;
	cin >> transamount;
	while (gettrans(transamount, daytransout, daytrans)) {
		cin >> transamount;
	}
	cout << "转账成功" << endl;
	balance -= transamount;
	daytransout += transamount;
	if (daytransout == daytrans) {
		cout << "今日转账限额已用尽" << endl;
	}
	else {
		cout << "今日剩余转账限额 " << daytrans - daytransout << endl;
	}
	save();
	showbalance();

}


int main() {
	
_RELOGIN:
	Person p;//创建对象
	int count = 3;//定义试错次数
	
	welcomevision();
	
	p.login();//登录
	
	functionvision();
	
	double daycash = 5000;
	double daytrans = 5000;
	double daytransout = 0;
	double dayout = 0;
	string choice01 = "0";
	double choice02;

		while (true) {
		_RECHOICE:
			std::cin >> choice01;
			stringstream ss;
			ss << choice01;
			ss >> choice02;
			while (true) {
				if (choice02 - (int)choice02 > 0 && choice02 - (int)choice02 < 1) {
					cout << "选择有误，请重新选择" << endl;
					goto _RECHOICE;
				}
				else {
					break;
				}
			}

			switch ((int)choice02) {
			case 1:
				p.showbalance();

				break;
			case 2:
				cout << "单次取款限额:2000元" << endl;
				cout << "单日取款限额:5000元" << endl;
				p.getcash(dayout, daycash);

				break;
			case 3:
				cout << "单次转账限额：2000元" << endl;
				cout << "单日转账限额：5000元" << endl;
				p.transfercash(daytransout, daytrans);

				break;
			case 4:
				p.changepassword();
				goto _RELOGIN;

				break;
			case 5:
				p.exitcard();

				return 0;
				break;
			default:
				cout << "选择有误，请重新选择" << endl;
				
				break;

			}
		}

	return 0;
}
