:-ensure_loaded("pokemon_list.pl").
:-ensure_loaded("pokemon_info_attacks.pl").
:-ensure_loaded("pokemon_route.pl").

player_starts(0,0).

get_element(0, [X|_], X).
get_element(N, [_|T], Y) :-
    N > 0,
    M is N - 1,
    get_element(M, T, Y).

% Esquerda
valid_neighbor(X, Y, M, NextX, NextY, Id, Level) :-
    NextX is X - 1, NextX >= 0, NextY is Y,
    get_element(NextY, M, Row),
    get_element(NextX, Row, (Id, Level)).

% Direita
valid_neighbor(X, Y, M, NextX, NextY, Id, Level) :-
    NextX is X + 1, NextY is Y,
    get_element(NextY, M, Row),
    get_element(NextX, Row, (Id, Level)).

% Baixo
valid_neighbor(X, Y, M, NextX, NextY, Id, Level) :-
    NextX is X, NextY is Y + 1,
    get_element(NextY, M, Row),
    get_element(NextX, Row, (Id, Level)).

% Cima
valid_neighbor(X, Y, M, NextX, NextY, Id, Level) :-
    NextX is X, NextY is Y - 1, NextY >= 0,
    get_element(NextY, M, Row),
    get_element(NextX, Row, (Id, Level)).

get_room(X, Y, M, [Id, Name, Level, NextX, NextY, Types]) :-
    valid_neighbor(X, Y, M, NextX, NextY, Id, Level),
    pokemon(Id, Name, Types).

% Retorna a lista dos quartos viáveis usando a formatação exigida pelo enunciado
next_rooms(X, Y, Rooms) :-
    route(M),
    findall(Room, get_room(X, Y, M, Room), Rooms).