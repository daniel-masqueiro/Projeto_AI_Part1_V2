import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

'''
Cria um sistema Fuzzy que recebe como input a diferença dos niveis
e o efeito do ataque e devolve como input a probabilidade de ganhar
'''
# TO DO-Feito

level_input = ctrl.Antecedent(np.arange(-10, 11, 1), 'level_input')
level_input['Muito Abaixo'] = fuzz.trapmf(level_input.universe, [-10, -10, -7, -2])
level_input['Equilibrado'] = fuzz.trimf(level_input.universe, [-5, 0, 5])
level_input['Muito Alto'] = fuzz.trapmf(level_input.universe, [2, 7, 10, 10])

effect_input = ctrl.Antecedent(np.arange(0, 2.1, 0.1), 'effect_input')
effect_input['Não Eficaz'] = fuzz.trapmf(effect_input.universe, [0, 0, 0.2, 0.5])
effect_input['Neutro'] = fuzz.trimf(effect_input.universe, [0.5, 1, 2])
effect_input['Eficaz'] = fuzz.trapmf(effect_input.universe, [1.5, 2, 2, 2])

win_prob = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'win_prob')
win_prob['Baixa'] = fuzz.trimf(win_prob.universe, [0, 0, 0.5])
win_prob['Média'] = fuzz.trimf(win_prob.universe, [0, 0.5, 1])
win_prob['Alta'] = fuzz.trimf(win_prob.universe, [0.5, 1, 1])

regra1 = ctrl.Rule(level_input['Muito Abaixo'] & effect_input['Não Eficaz'], win_prob['Baixa'])
regra2 = ctrl.Rule(level_input['Muito Abaixo'] & effect_input['Neutro'], win_prob['Baixa'])
regra3 = ctrl.Rule(level_input['Muito Abaixo'] & effect_input['Eficaz'], win_prob['Média'])
regra4 = ctrl.Rule(level_input['Equilibrado'] & effect_input['Não Eficaz'], win_prob['Baixa'])
regra5 = ctrl.Rule(level_input['Equilibrado'] & effect_input['Neutro'], win_prob['Média'])
regra6 = ctrl.Rule(level_input['Equilibrado'] & effect_input['Eficaz'], win_prob['Alta'])
regra7 = ctrl.Rule(level_input['Muito Alto'] & effect_input['Não Eficaz'], win_prob['Média'])
regra8 = ctrl.Rule(level_input['Muito Alto'] & effect_input['Neutro'], win_prob['Alta'])
regra9 = ctrl.Rule(level_input['Muito Alto'] & effect_input['Eficaz'], win_prob['Alta'])

sistema_combate = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5, regra6, regra7, regra8, regra9])
simulacao = ctrl.ControlSystemSimulation(sistema_combate)


def calculate_prob(level_input, effect_input):
    global_level_input = globals()['level_input']
    global_effect_input = globals()['effect_input']

    l_pos = np.where(global_level_input.universe == level_input)[0][0]
    e_pos = np.where(global_effect_input.universe == effect_input)[0][0]

    l_abaixo = global_level_input['Muito Abaixo'].mf[l_pos]
    l_eq = global_level_input['Equilibrado'].mf[l_pos]
    l_alto = global_level_input['Muito Alto'].mf[l_pos]

    e_nao_eficaz = global_effect_input['Não Eficaz'].mf[e_pos]
    e_neutro = global_effect_input['Neutro'].mf[e_pos]
    e_eficaz = global_effect_input['Eficaz'].mf[e_pos]

    r1_contrib = np.fmin(min(l_abaixo, e_nao_eficaz), win_prob['Baixa'].mf)
    r2_contrib = np.fmin(min(l_abaixo, e_neutro), win_prob['Baixa'].mf)
    r3_contrib = np.fmin(min(l_abaixo, e_eficaz), win_prob['Média'].mf)

    r4_contrib = np.fmin(min(l_eq, e_nao_eficaz), win_prob['Baixa'].mf)
    r5_contrib = np.fmin(min(l_eq, e_neutro), win_prob['Média'].mf)
    r6_contrib = np.fmin(min(l_eq, e_eficaz), win_prob['Alta'].mf)

    r7_contrib = np.fmin(min(l_alto, e_nao_eficaz), win_prob['Média'].mf)
    r8_contrib = np.fmin(min(l_alto, e_neutro), win_prob['Alta'].mf)
    r9_contrib = np.fmin(min(l_alto, e_eficaz), win_prob['Alta'].mf)

    else_link = np.fmax(r1_contrib, np.fmax(r2_contrib, np.fmax(r3_contrib, np.fmax(r4_contrib, np.fmax(r5_contrib,
                                                                                                        np.fmax(
                                                                                                            r6_contrib,
                                                                                                            np.fmax(
                                                                                                                r7_contrib,
                                                                                                                np.fmax(
                                                                                                                    r8_contrib,
                                                                                                                    r9_contrib))))))))

    prob_final = fuzz.defuzz(win_prob.universe, else_link, 'centroid')

    return prob_final

    


