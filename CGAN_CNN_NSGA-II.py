import itertools
import matplotlib
from joblib import load
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import time

print(time.strftime("%H:%M:%S", time.localtime()))


def calculate_pv_cost(total_years):
    total_build = 0.0
    total_maintain = 0.0

    for year in range(total_years):
        build_cost = max(3.4 - (year // 3) * 0.1, 0)
        maintain_cost = max(0.045 - year * 0.001, 0)

        total_build += build_cost
        total_maintain += maintain_cost

    avg_build = total_build / total_years
    avg_maintain = total_maintain / total_years

    return avg_build * 1000, avg_maintain * 1000


class CNNEmissionsPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 16, 5, padding=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 11, 11),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        return self.fc(x.view(x.size(0), -1))


class PVOptimizationProblem(Problem):
    def __init__(self, years, u, cost_green, cost_non, production_bounds, PV_max):
        n_var = 13
        n_obj = 3
        n_constr = 3

        xl = production_bounds[:, 0].tolist() + [0.0, 0.0]
        xu = production_bounds[:, 1].tolist() + [1.0, 1.0]

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.years = years
        self.u = np.array(u)
        self.cost_green = cost_green
        self.cost_non = cost_non
        self.PV_max = PV_max

        self.x_scaler = load('final_model_result/finetune_x_scaler.pkl')
        self.y_scaler = load('final_model_result/finetune_y_scaler.pkl')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = CNNEmissionsPredictor().to(self.device)
        self.cnn_model.load_state_dict(
            torch.load('final_model_result/finetune_emission_predictor.pth', map_location=self.device))
        self.cnn_model.eval()

    def _evaluate(self, X, out, *args, **kwargs):
        p = X[:, :11]
        f_green = X[:, 11]
        f_non = X[:, 12]
        f_PV = 1.0 - f_green - f_non

        total_production = np.sum(p, axis=1)

        industry_electricity = p * self.u

        scaled_electricity = self.x_scaler.transform(industry_electricity)
        electricity_tensor = torch.tensor(scaled_electricity, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            emissions_scaled = self.cnn_model(electricity_tensor)
            emissions = self.y_scaler.inverse_transform(emissions_scaled.cpu().numpy())
            f2 = np.sum(emissions, axis=1)

        total_electricity = np.sum(industry_electricity, axis=1)

        PV_elec = total_electricity * f_PV

        total_pv_capacity = PV_elec / 2500
        existing_pv_capacity = 95345.65
        new_pv_capacity = np.maximum(total_pv_capacity - existing_pv_capacity, 0.0)
        pv_build_cost, pv_maintain_cost = calculate_pv_cost(self.years)

        annualized_build_cost = (new_pv_capacity * pv_build_cost) / 20

        annual_maintain_cost = (existing_pv_capacity + new_pv_capacity) * pv_maintain_cost

        f1 = -total_production
        f3 = (total_electricity * f_green * self.cost_green) + \
             (total_electricity * f_non * self.cost_non) + \
             annualized_build_cost + annual_maintain_cost

        out["F"] = np.column_stack([f1, f2, f3])

        g1 = f_green + f_non - 1.0
        g2 = PV_elec - self.PV_max
        g3 = f_green - 0.15
        out["G"] = np.column_stack([g1, g2, g3])


years = 2030 - 2023
u = np.array([3410194.98, 1151508.923, 471677.7319, 2438320.632, 1946626.351,
              9746860.234, 97958.04664, 11738661.29, 2738530.153, 2040604.155, 3793394.982])
cost_green = 0.397 - 0.02
cost_non = 0.41
production_bounds = np.array([[1484.2, 2968.8], [82.7, 165.4], [274.4, 548.8], [47.8, 95.6],
                              [18, 36], [238.3, 476.6], [87.9, 175.8], [24.9, 49.8],
                              [76.5, 153], [4.1, 8.2], [163.1, 326.2]])
PV_max = 135e8

problem = PVOptimizationProblem(years, u, cost_green, cost_non, production_bounds, PV_max)

pop_sizes = [50, 100, 200]
crossover_probs = [0.8, 0.85, 0.9]
crossover_etas = [15, 20, 25]
mutation_probs = [0.01, 0.05, 0.1]
mutation_etas = [20, 30, 40]

param_combinations = list(itertools.product(pop_sizes, crossover_probs, crossover_etas,
                                            mutation_probs, mutation_etas))

ref_point = np.array([1.1, 1.1, 1.1])

best_hv = 0
best_params = None

for params in param_combinations:
    pop_size, crossover_prob, crossover_eta, mutation_prob, mutation_eta = params
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=crossover_eta),
        mutation=PM(prob=mutation_prob, eta=mutation_eta),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", 1000)
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    F_normalized = (res.F - res.F.min(axis=0)) / (res.F.max(axis=0) - res.F.min(axis=0) + 1e-6)
    hv_indicator = HV(ref_point=ref_point)
    hv = hv_indicator(F_normalized)
    if hv > best_hv:
        best_hv = hv
        best_params = params

print(f"最优参数: pop_size={best_params[0]}, crossover_prob={best_params[1]}, "
      f"crossover_eta={best_params[2]}, mutation_prob={best_params[3]}, "
      f"mutation_eta={best_params[4]}")
print(f"Hypervolume: {best_hv:.4f}")

algorithm = NSGA2(
    pop_size=best_params[0],
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=best_params[1], eta=best_params[2]),
    mutation=PM(prob=best_params[3], eta=best_params[4]),
    eliminate_duplicates=True,
)
termination = get_termination("n_gen", 3000)
res = minimize(problem, algorithm, termination, seed=1, verbose=True)

print("\nPareto前沿解及对应的生产和用电方案：")
solutions = res.X
F_values = res.F
existing_pv_capacity = 95345.65

for i, (sol, obj) in enumerate(zip(solutions, F_values)):
    p_values = sol[:11]
    f_green_val = sol[11]
    f_non_val = sol[12]
    f_PV_val = 1.0 - f_green_val - f_non_val

    total_elec = np.sum(p_values * np.array(u))
    PV_elec = total_elec * f_PV_val

    total_pv_capacity = PV_elec / 2500
    new_pv_capacity = max(total_pv_capacity - existing_pv_capacity, 0.0)
    total_pv_capacity_actual = existing_pv_capacity + new_pv_capacity

    print("====================================")
    print(f"解 {i + 1}:")
    print("11个行业产值：（亿元）", np.around(p_values, 2))
    print("绿电比例：{:.3f}，非绿电比例：{:.3f}，光伏比例：{:.3f}".format(f_green_val, f_non_val, f_PV_val))
    print("总用电量：{:.2f} kWh".format(total_elec))
    print("光伏发电量：{:.2f} kWh (需装机容量：{:.2f} kW)".format(PV_elec, total_pv_capacity))
    print("新增装机容量：{:.2f} kW，总装机容量：{:.2f} kW".format(new_pv_capacity, total_pv_capacity_actual))
    print(f"目标函数值：产值={-obj[0]:.2f}亿元, 碳排放={obj[1]:.2f} 万tCO2, 总成本={obj[2]:.2f}元")

matplotlib.rc("font", family='Microsoft YaHei')
F_plot = res.F * [-1, 1, 1]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(F_plot[:, 0], F_plot[:, 1], F_plot[:, 2], c='blue', s=30)
plt.title("Pareto Front")
plt.show()


matplotlib.rc("font", family='Microsoft YaHei')


def calculate_hypervolume(F):
    F = np.asarray(F_normalized)
    ref_point = np.array([1.1, 1.1, 1.1])
    hv_indicator = HV(ref_point=ref_point)
    hv = hv_indicator(F)
    return hv


def calculate_crowding_distance(F):
    n_points, n_obj = F.shape
    crowding = np.zeros(n_points)

    for i in range(n_obj):
        idx = F[:, i].argsort()
        crowding[idx[[0, -1]]] = np.inf

        norm = F[idx[-1], i] - F[idx[0], i]
        if norm < 1e-6:
            continue

        crowding[idx[1:-1]] += (F[idx[2:], i] - F[idx[:-2], i]) / norm

    return crowding


obj_min = res.F.min(axis=0)
obj_max = res.F.max(axis=0)
F_normalized = (res.F - obj_min) / (obj_max - obj_min)

F_positive = res.F.copy()
F_positive[:, 0] = -F_positive[:, 0]
obj_min = F_positive.min(axis=0)
obj_max = F_positive.max(axis=0)

hypervolume = calculate_hypervolume(F_normalized)

crowding = calculate_crowding_distance(F_normalized)

valid_crowding = crowding[np.isfinite(crowding)]

if len(valid_crowding) > 0:
    min_crowding = valid_crowding.min()
    max_crowding = valid_crowding.max()
    avg_crowding = valid_crowding.mean()
else:
    min_crowding = max_crowding = avg_crowding = np.nan
violations = np.sum(res.G > 0, axis=0)
feasible_ratio = np.mean(np.all(res.G <= 0, axis=1))

print("\n============= 优化结果评估报告 =============")
print(f"帕累托解数量：{len(res.X)}")
print(f"目标范围（最小值）：产值={obj_min[0]:.2f}亿元, 碳排放={obj_min[1]:.2f}万tCO2, 年化成本={obj_min[2]:.2f}元")
print(f"目标范围（最大值）：产值={obj_max[0]:.2f}亿元, 碳排放={obj_max[1]:.2f}万tCO2, 年化成本={obj_max[2]:.2f}元")
print(f"Hypervolume指标：{best_hv:.4f}（越接近1.331越好）")
if not np.isnan(avg_crowding):
    print(f"拥挤距离范围：[{min_crowding:.2f}, {max_crowding:.2f}]，平均：{avg_crowding:.2f}")
else:
    print("拥挤距离：所有解均为边界点（无限大）")
print(f"约束违反情况：{violations}（应全为0）")
print(f"可行解占比：{feasible_ratio * 100:.2f}%")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ["产值（亿元）", "碳排放（万tCO2）", "年化成本（元）"]
for i in range(3):
    axes[i].hist(F_positive[:, i], bins=20, edgecolor='k')
    axes[i].set_title(titles[i])
plt.tight_layout()
plt.show()

electric_data = []
result_data = []
output_lines = []

existing_pv_capacity = 95345.65

for i, (sol, obj) in enumerate(zip(solutions, F_values)):
    p_values = sol[:11]
    f_green_val = sol[11]
    f_non_val = sol[12]
    f_PV_val = 1.0 - f_green_val - f_non_val

    total_elec = np.sum(p_values * np.array(u))
    PV_elec = total_elec * f_PV_val

    total_pv_capacity = PV_elec / 2500
    new_pv_capacity = max(total_pv_capacity - existing_pv_capacity, 0.0)
    total_pv_capacity_actual = existing_pv_capacity + new_pv_capacity

    electric_per_industry = p_values * u
    electric_data.append([i+1] + electric_per_industry.tolist())

    result_row = {'解编号': i+1}
    for j in range(11):
        result_row[f'行业{j+1}产值'] = p_values[j]
    result_row.update({
        '绿电比例': f_green_val,
        '非绿电比例': f_non_val,
        '光伏比例': f_PV_val,
        '总用电量': total_elec,
        '光伏发电量': PV_elec,
        '总装机容量': total_pv_capacity_actual,
        '新增装机容量': new_pv_capacity,
        '产值': -obj[0],
        '碳排放': obj[1],
        '总成本': obj[2]
    })
    result_data.append(result_row)

    output_lines.append("====================================")
    output_lines.append(f"解 {i + 1}:")
    output_lines.append("11个行业产值（亿元）: " + str(np.around(p_values, 2)))
    output_lines.append(f"绿电比例: {f_green_val:.3f}, 非绿电比例: {f_non_val:.3f}, 光伏比例: {f_PV_val:.3f}")
    output_lines.append(f"总用电量: {total_elec:.2f} kWh")
    output_lines.append(f"光伏发电量: {PV_elec:.2f} kWh (需装机容量: {total_pv_capacity:.2f} kW)")
    output_lines.append(f"新增装机容量: {new_pv_capacity:.2f} kW，总装机容量: {total_pv_capacity_actual:.2f} kW")
    output_lines.append(f"目标函数值: 产值={-obj[0]:.2f}亿元, 碳排放={obj[1]:.2f} 万tCO2, 总成本={obj[2]:.2f}元")

electric_columns = ['解编号'] + [f'行业{i+1}用电量' for i in range(11)]
df_electric = pd.DataFrame(electric_data, columns=electric_columns)
df_electric.to_excel('final_result/electric.xlsx', index=False)

df_result = pd.DataFrame(result_data)

report_data = {
    '指标': ['帕累托解数量', '目标范围（最小值）', '目标范围（最大值）', 'Hypervolume指标',
           '拥挤距离最小值', '拥挤距离最大值', '拥挤距离平均值', '约束违反情况', '可行解占比'],
    '值': [
        len(res.X),
        f"产值={obj_min[0]:.2f}亿元, 碳排放={obj_min[1]:.2f}万tCO2, 年化成本={obj_min[2]:.2f}元",
        f"产值={obj_max[0]:.2f}亿元, 碳排放={obj_max[1]:.2f}万tCO2, 年化成本={obj_max[2]:.2f}元",
        f"{hypervolume:.4f}",
        f"{min_crowding:.2f}" if not np.isnan(min_crowding) else "N/A",
        f"{max_crowding:.2f}" if not np.isnan(max_crowding) else "N/A",
        f"{avg_crowding:.2f}" if not np.isnan(avg_crowding) else "N/A",
        str(violations),
        f"{feasible_ratio * 100:.2f}%"
    ]
}
df_report = pd.DataFrame(report_data)

df_output = pd.DataFrame({'原始输出内容': output_lines})

with pd.ExcelWriter('final_result/result.xlsx') as writer:
    df_result.to_excel(writer, sheet_name='解的详细数据', index=False)
    df_report.to_excel(writer, sheet_name='评估报告', index=False)

print(time.strftime("%H:%M:%S", time.localtime()))