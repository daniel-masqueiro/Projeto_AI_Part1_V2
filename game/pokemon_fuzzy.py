import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

level_input = ctrl.Antecedent(np.arange(-10, 11, 1), 'level_input')
effect_input = ctrl.Antecedent(np.arange(0, 4.25, 0.25), 'effect_input')
win_prob = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'win_prob')

level_input['Muito Abaixo'] = fuzz.trapmf(level_input.universe, [-10, -10, -5, -1])
level_input['Equilibrado'] = fuzz.trimf(level_input.universe, [-3, 0, 3])
level_input['Muito Alto'] = fuzz.trapmf(level_input.universe, [1, 5, 10, 10])

effect_input['Não Eficaz'] = fuzz.trapmf(effect_input.universe, [0, 0, 0.25, 0.5])
effect_input['Neutro'] = fuzz.trimf(effect_input.universe, [0.5, 1, 2])
effect_input['Eficaz'] = fuzz.trapmf(effect_input.universe, [1.5, 2, 4, 4])

win_prob['Baixa'] = fuzz.trimf(win_prob.universe, [0, 0, 0.5])
win_prob['Média'] = fuzz.trimf(win_prob.universe, [0, 0.5, 1])
win_prob['Alta'] = fuzz.trimf(win_prob.universe, [0.5, 1, 1])

def calculate_prob(level_val, effect_val):
    level_val = max(-10, min(10, int(level_val)))
    effect_val = max(0.0, min(4.0, effect_val))

    effect_val = round(effect_val * 4) / 4.0

    l_pos = np.where(level_input.universe == level_val)[0][0]
    e_pos = np.where(effect_input.universe == effect_val)[0][0]

    l_abaixo = level_input['Muito Abaixo'].mf[l_pos]
    l_eq = level_input['Equilibrado'].mf[l_pos]
    l_alto = level_input['Muito Alto'].mf[l_pos]

    e_nao_eficaz = effect_input['Não Eficaz'].mf[e_pos]
    e_neutro = effect_input['Neutro'].mf[e_pos]
    e_eficaz = effect_input['Eficaz'].mf[e_pos]

    r1_contrib = np.fmin(min(l_abaixo, e_nao_eficaz), win_prob['Baixa'].mf)
    r2_contrib = np.fmin(min(l_abaixo, e_neutro), win_prob['Baixa'].mf)
    r3_contrib = np.fmin(min(l_abaixo, e_eficaz), win_prob['Média'].mf)

    r4_contrib = np.fmin(min(l_eq, e_nao_eficaz), win_prob['Baixa'].mf)
    r5_contrib = np.fmin(min(l_eq, e_neutro), win_prob['Média'].mf)
    r6_contrib = np.fmin(min(l_eq, e_eficaz), win_prob['Alta'].mf)

    r7_contrib = np.fmin(min(l_alto, e_nao_eficaz), win_prob['Média'].mf)
    r8_contrib = np.fmin(min(l_alto, e_neutro), win_prob['Alta'].mf)
    r9_contrib = np.fmin(min(l_alto, e_eficaz), win_prob['Alta'].mf)

    else_link = np.fmax.reduce([
        r1_contrib, r2_contrib, r3_contrib,
        r4_contrib, r5_contrib, r6_contrib,
        r7_contrib, r8_contrib, r9_contrib
    ])

    prob_final = fuzz.defuzz(win_prob.universe, else_link, 'centroid')

    return prob_final