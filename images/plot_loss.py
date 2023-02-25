import matplotlib.pyplot as plt

x = [[202.219186173195, 184.9899576220594, 179.82773087495065, 173.30282086368297, 171.4692370986613, 171.2819236892912, 169.92130864337616, 169.82843553781942, 169.15966040824014, 168.82823972316515, 168.49431077633437, 168.37262387782016, 168.29546071700236, 167.5847781304526, 167.2254830835744, 167.33205935510065, 166.76635005281491, 167.6528306673769, 167.4044344900783, 167.03199064061772],
[204.01296453154265, 187.11870362654898, 179.92371507923465, 175.7268755050556, 174.6348034835226, 173.62237235074014, 171.07676974376517, 170.68867522584037, 171.87566446093166, 170.49857198705593, 169.2111882271085, 168.95696069577033, 168.8230881413856, 168.19367282959416, 167.71698736597554, 167.57821074156206, 167.23890701496265, 167.06007832534755, 167.1981691892845, 166.95614005706898],
[205.02597327736802, 179.6620355840236, 174.56564416400676, 172.0220266062968, 169.23294764114485, 167.2246864352585, 166.30357136978955, 166.05853616516504, 165.23178943575584, 164.63735978544048, 164.4755851839526, 164.50526467231487, 163.47856668524054, 161.80603619240404, 161.41954221332998, 161.22286556305738, 160.97491996376112, 160.1981158597884, 160.3327551545515, 159.9073805101616],
[207.17533271138853, 183.74284204383142, 177.48492760243073, 174.85643549724796, 165.9905516991655, 165.89297376580535, 165.12805581412886, 164.78251872756812, 164.98453976448656, 164.92917192391826, 164.3512370450773, 164.86555068609442, 164.68216823917948, 164.01192547084955, 164.10736382044138, 164.29195287886262, 161.95290223125596, 161.36543469012466, 161.16139043366593, 160.22672656349062],
[202.71752209869453, 180.067488633244, 171.61221166474522, 168.73386596224867, 168.39862895870917, 168.01254945131916, 168.3231254809425, 166.33605675584337, 165.83777253387456, 166.21473768144782, 165.78991388675274, 165.6679418789758, 165.57700234758858, 162.96213620637565, 161.30562750089163, 161.3271336129849, 160.49888891153347, 159.75350268288983, 159.5205109617516, 159.43740291974805],
[206.44758421336945, 188.2822031683629, 186.0797162250785, 184.41976667133818, 179.0419979330607, 176.9117199097936, 172.44250435522213, 169.73208448992898, 167.3854466520174, 166.8803037652715, 166.39278621626133, 166.14503060011504, 165.98245563648058, 163.96742474909357, 163.84955366790712, 163.48016755953233, 163.40103521950797, 163.14247171788008, 162.92491609959475, 162.70088340198708],
[212.26446079018913, 188.45985851626085, 184.4651337316726, 178.91600454886498, 177.5405185631882, 176.05649760610282, 174.95394634167008, 172.60264642102408, 170.9534817325003, 171.63245521176532, 170.76595801740956, 169.7014292251605, 168.7536882188191, 168.03790760218834, 167.50494208305122, 166.7945424760324, 166.242125213941, 165.613422255865, 165.57463027455208, 165.5162968453374],
[195.5909864424586, 185.9639921075606, 170.98324398638525, 169.58842714486275, 167.7545242502863, 165.75195859882635, 165.5194788460607, 165.45440779589137, 164.36034113501682, 163.87645612522869, 163.7330816110511, 162.90435216104663, 162.67419904905006, 162.11812490442105, 161.81026538619557, 161.6956265090401, 161.5912185720386, 161.23256139194433, 161.17803375624635, 161.21974051583626],
[208.42633676482666, 186.6654378869033, 182.01469012887617, 177.10576711083826, 173.74005880847156, 166.97679195228255, 165.33713850132435, 165.62508649428648, 164.80192756246973, 164.40223957202394, 163.4508096681667, 162.53311526324785, 162.4304716503243, 162.69344951972312, 162.64620606948603, 162.33133455983267, 161.56386587468904, 161.6929581446309, 160.97972198450196, 160.95521152430092],
[203.28736092043005, 189.42435800638162, 180.44560620226113, 176.11400977436796, 173.55330582892032, 171.72129462982184, 170.70939058704639, 170.14187135042295, 166.48515707610665, 166.63545445926516, 166.04724926065268, 165.09061416330707, 164.8583123042507, 164.30951408656352, 163.94987431003, 163.84428188998052, 163.23182673402786, 162.4139911905641, 162.24482623064944, 162.17973273737266]
]

plt.figure(figsize=(8, 5))
y = [i for i in range(len(x[0]))]
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (0, 0, 0)]
for i in range(len(x)):
    plt.plot(x[i], y, color=colors[i])
plt.grid()
plt.xlabel("rounds")
plt.ylabel("loss")
plt.title("loss curves of XGBoosts in a Bagging")
plt.savefig("loss_curves")
plt.show()